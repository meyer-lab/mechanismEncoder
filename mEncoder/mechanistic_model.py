import sympy as sp
import re

from typing import Iterable, Optional, Dict, Tuple, Union
from pysb import Monomer, Expression, Parameter, Rule, Model, Observable


def generate_pathway(model: Model,
                     proteins: Iterable[Tuple[str, Dict[str, Iterable[str]]]]):
    """
    Adds synthesis and phospho-signal transduction rules to the model
    based on the input specifications

    :param model:
        model to which rules will be added

    :param proteins:
        pathway specification
    """
    for p_name, site_activators in proteins:
        add_monomer_synth_deg(p_name, psites=site_activators.keys())
        for site, activators in site_activators.items():
            add_activation(model, p_name, site, 'phosphorylation', activators)


def add_monomer_synth_deg(m_name: str,
                          psites: Optional[Iterable[str]] = None,
                          nsites: Optional[Iterable[str]] = None,
                          asites: Optional[Iterable[str]] = None,):
    """
    Adds the respective monomer plus synthesis rules and basal
    activation/deactivation rules for all activateable sites

    :param m_name:
        monomer name

    :param psites:
        phospho sites

    :param nsites:
        nucleotide sites

    :param asites:
        other activity encoding sites
    """

    if psites is None:
        psites = []
    else:
        psites = [
            site
            for psite in psites
            for site in psite.split('_')
        ]

    if nsites is None:
        nsites = []

    if asites is None:
        asites = []

    sites = psites + nsites + asites

    m = Monomer(
        m_name, sites=[psite for psite in sites],
        site_states={
            site: ['u', 'p'] if site in psites
            else ['gdp', 'gtp'] if site in nsites
            else ['inactive', 'active']
            for site in sites
        }
    )

    kdeg = Parameter(f'{m_name}_degradation_kdeg', 1.0)
    t = Parameter(f'{m_name}_eq', 100.0)
    ksyn = Expression(f'{m_name}_synthesis_ksyn', t*kdeg)
    syn_rate = Expression(f'{m_name}_synthesis_rate',
                          ksyn * get_autoencoder_modulator(t))

    Rule(f'synthesis_{m_name}', None >> m(
        **{site:
           'u' if site in psites
           else 'gdp' if site in nsites
           else 'inactive'
           for site in sites}
    ), syn_rate)

    deg_rate = Expression(f'{m_name}_degradation_rate', kdeg)
    Rule(f'degradation_{m_name}', m() >> None, deg_rate)

    # basal activation
    for sites, labels, fstate, rstate in zip(
        [psites, nsites, asites],
        [('phosphorylation', 'dephosphorylation'),
         ('gtp_exchange', 'gdp_exchange'),
         ('activation', 'deactivation')],
        ['p', 'gtp', 'active'],
        ['u', 'gdp', 'indactive']
    ):
        for site in sites:
            kcats = [
                Parameter(f'{m_name}_{label}_{site}_base_kcat', 1.0)
                for label in labels
            ]
            rates = [
                Expression(f'{m_name}_{label}_{site}_base_rate',
                           kcat * get_autoencoder_modulator(kcat))
                for kcat, label in zip(kcats, labels)
            ]

            Rule(F'{m_name}_{site}_base',
                 m(**{site: fstate}) | m(**{site: rstate}), *rates)

    return m


def add_or_get_modulator_obs(model: Model, modulator: str):
    """
    Adds an observable to the model that tracks the specified modulator

    :param model:
        model to which the observable will be added

    :param modulator:
        string definition of an observable in format
        `{monomer_name}__{site}_{site_condition}`
    """
    mod_name = f'{modulator}_obs'
    if mod_name in model.observables.keys():
        modulator_obs = model.observables[f'{modulator}_obs']
    else:
        desc = modulator.split('__')
        mono_name = desc[0]
        if len(desc) > 1:
            site_conditions = desc[1:]
        else:
            site_conditions = []

        try:
            site_conditions = {
                cond.split('_')[0]: cond.split('_')[1]
                for cond in site_conditions
            }
        except IndexError:
            raise ValueError(f'Malformed site condition {site_conditions}')

        modulator_obs = Observable(mod_name,
                                   model.components[mono_name](
                                       **site_conditions
                                   ))

    return modulator_obs


def add_activation(
        model: Model, m_name: str, site: str, activation_type: str,
        activators: Optional[Iterable[str]] = None,
        deactivators: Optional[Iterable[str]] = None
):
    """
    Adds activation/deactivation rules to a specific site

    :param model:
        model to which the rules will be added

    :param m_name:
        monomer name

    :param site:
        site name

    :param activation_type:
        type of activation, valid values are
        {`phosphorylation`, `nucleotide_exchange`, `tf_activation`}

    :param activators:
        molecular species that activate the respective site, format
        according to modulator format in :py:func:`add_or_get_modulator_obs`

    :param deactivators:
        molecular species that deactivate the respective site, format
        according to modulator format in :py:func:`add_or_get_modulator_obs`

    """

    if activators is None:
        activators = []

    if deactivators is None:
        deactivators = []

    if m_name not in model.monomers.keys():
        raise ValueError(f'{m_name} is not a monomer in the model.')

    mono = model.monomers[m_name]

    if activation_type == 'phosphorylation':
        valid_states = ['u', 'p']
    elif activation_type == 'nucleotide_exchange':
        valid_states = ['gdp', 'gtp']
    elif activation_type == 'tf_activation':
        valid_states = ['active', 'inactive']
    else:
        raise ValueError(f'Invalid activation type {activation_type}.')

    sites = [site for site in site.split('_')]

    for site in sites:
        if site not in mono.site_states \
                or len(mono.site_states[site]) != len(valid_states) \
                or any(state not in mono.site_states[site]
                       for state in valid_states):
            raise ValueError(f'{site} is not a valid target for '
                             f'{activation_type}.')

    if activation_type == 'phosphorylation':
        forward = 'phosphorylation'
        backward = 'dephosphorylation'
        fstate = {site: 'u' for site in sites}
        rstate = {site: 'p' for site in sites}
    elif activation_type == 'nucleotide_exchange':
        forward = 'gtp_exchange'
        backward = 'gdp_exchange'
        fstate = {site: 'gdp' for site in sites}
        rstate = {site: 'gtp' for site in sites}
    else:
        raise ValueError(f'Invalid activation type {activation_type}.')

    for label, educts, products, modulators in zip(
            [forward,        backward],
            [mono(**fstate), mono(**rstate)],
            [mono(**rstate), mono(**fstate)],
            [activators, deactivators],
    ):
        for modulator in modulators:
            kcat = Parameter(f'{m_name}_{label}_{site}_{modulator}_kcat', 1.0)
            rate = Expression(f'{m_name}_{label}_{site}_{modulator}_rate',
                              kcat
                              * add_or_get_modulator_obs(model, modulator)
                              * get_autoencoder_modulator(kcat))

            Rule(F'{m_name}_{label}_{site}_{modulator}', educts >> products,
                 rate)


def get_autoencoder_modulator(par: Parameter):
    """
    Generate a new expression that allows modulation of a rate according to
    input parameter. Applies a sigmoid transformation.
    """
    return Parameter(f'INPUT_{par.name}', 0.0)


def add_abundance_observables(model):
    """
    Adds an observable that tracks the normalized absolute abundance of a
    protein
    """
    for monomer in model.monomers:
        obs = Observable(f'total_{monomer.name}', monomer())
        scale = Parameter(f't{monomer.name}_scale', 1.0)
        offset = Parameter(f't{monomer.name}_offset', 1.0)
        Expression(f't{monomer.name}_obs', sp.log(scale * (obs + offset)))


def add_phospho_observables(model):
    """
    Adds an observable that tracks the normalized absolute abundance of a
    phosphorylated site
    """
    for monomer in model.monomers:
        for site in monomer.site_states:
            if re.match(r'[YTS][0-9]+$', site):
                obs = Observable(f'p{monomer.name}_{site}',
                                 monomer(**{site: 'p'}))
                scale = Parameter(f'p{monomer.name}_{site}_scale', 1.0)
                offset = Parameter(f'p{monomer.name}_{site}_offset', 1.0)
                Expression(f'p{monomer.name}_{site}_obs',
                           sp.log(scale * (obs + offset)))




