import re
import itertools as itt

from typing import Iterable, Optional, Dict, Tuple, List
from pysb import (
    Monomer, Expression, Parameter, Rule, Model, Observable, Initial
)


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

    for p_name, site_activators in proteins:
        for site, activators in site_activators.items():
            add_activation(model, p_name, site, 'phosphorylation', activators)


def add_monomer_synth_deg(m_name: str,
                          psites: Optional[Iterable[str]] = None,
                          nsites: Optional[Iterable[str]] = None,
                          asites: Optional[Iterable[str]] = None,
                          asite_states: Optional[Iterable[str]] = None):
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
        psites = list({
            site
            for psite in psites
            for site in psite.split('_')
        })

    if nsites is None:
        nsites = []

    if asites is None:
        asites = []

    if asite_states is None:
        asite_states = ['inactive', 'active']

    sites = psites + nsites + asites + ['inh']

    m = Monomer(
        m_name, sites=sites,
        site_states={
            site: ['u', 'p'] if site in psites
            else ['gdp', 'gtp'] if site in nsites
            else asite_states
            for site in sites
            if site in psites + nsites + asites
        }
    )

    kdeg = Parameter(f'{m_name}_degradation_kdeg', 1.0)
    t = Parameter(f'{m_name}_eq', 100.0)
    ksyn = Expression(f'{m_name}_synthesis_ksyn', t*kdeg)
    syn_rate = Expression(f'{m_name}_synthesis_rate',
                          ksyn * get_autoencoder_modulator(t))

    syn_prod = m(
        **{site:
           'u' if site in psites
           else 'gdp' if site in nsites
           else None if site is 'inh'
           else asite_states[0]
           for site in sites}
    )

    Rule(f'synthesis_{m_name}', None >> syn_prod, syn_rate)
    deg_rate = Expression(f'{m_name}_degradation_rate', kdeg)
    Rule(f'degradation_{m_name}', m() >> None, deg_rate)
    t_ss = Expression(f'{m_name}_ss', syn_rate/deg_rate)
    Initial(syn_prod, t_ss)


    # basal activation
    for sites, labels, fstate, rstate in zip(
        [psites, nsites, asites],
        [('phosphorylation', 'dephosphorylation'),
         ('gtp_exchange', 'gdp_exchange'),
         ('activation', 'deactivation')],
        ['p', 'gtp', asite_states[1]],
        ['u', 'gdp', asite_states[0]]
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
        mono_name, site_conditions = site_states_from_string(modulator)

        # uninhibited
        site_conditions['inh'] = None

        modulator_obs = Observable(mod_name,
                                   model.components[mono_name](
                                       **site_conditions
                                   ))

    return modulator_obs


def site_states_from_string(obs_string):
    desc = obs_string.split('__')
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

    return mono_name, site_conditions


def add_activation(
        model: Model, m_name: str, site: str, activation_type: str,
        activators: Optional[Iterable[str]] = None,
        deactivators: Optional[Iterable[str]] = None,
        site_states: Optional[Iterable[str]] = None,
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

    if site_states is not None:
        valid_states = site_states
    elif activation_type == 'phosphorylation':
        valid_states = ['u', 'p']
    elif activation_type == 'nucleotide_exchange':
        valid_states = ['gdp', 'gtp']
    elif activation_type == 'tf_activation':
        valid_states = ['inactive', 'active']
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
    elif activation_type == 'nucleotide_exchange':
        forward = 'gtp_exchange'
        backward = 'gdp_exchange'
    elif activation_type == 'activation':
        forward = 'gtp_exchange'
        backward = 'gdp_exchange'
    else:
        raise ValueError(f'Invalid activation type {activation_type}.')
    fstate = {site: valid_states[0] for site in sites}
    rstate = {site: valid_states[1] for site in sites}

    for label, educts, products, modulators in zip(
            [forward,        backward],
            [mono(**fstate), mono(**rstate)],
            [mono(**rstate), mono(**fstate)],
            [activators, deactivators],
    ):
        for modulator in modulators:
            kcat = Parameter(f'{m_name}_{label}_{site}_{modulator}_kcat', 1.0)
            rate_expr = kcat * add_or_get_modulator_obs(model, modulator)
            # if label == forward:
            rate_expr *= get_autoencoder_modulator(kcat)
            rate = Expression(f'{m_name}_{label}_{site}_{modulator}_rate',
                              rate_expr)

            Rule(F'{m_name}_{label}_{site}_{modulator}', educts >> products,
                 rate)


def get_autoencoder_modulator(par: Parameter):
    """
    Generate a new expression that allows modulation of a rate according to
    input parameter. Applies a sigmoid transformation.
    """
    return Parameter(f'INPUT_{par.name}', 0.0)


def add_observables(model: Model):
    """
    Adds a observable that tracks the normalized absolute abundance of all
    phosphorylated site combinations for all monomers
    """
    for monomer in model.monomers:
        Observable(f't{monomer.name}', monomer())
        psites = [site for site in monomer.site_states.keys()
                  if re.match(r'[YTS][0-9]+$', site)]
        for nsites in range(1, len(psites)+1):
            for sites in itt.combinations(psites, nsites):
                sites = sorted(sites)
                Observable(f'p{monomer.name}_{"_".join(sites)}',
                           monomer(**{site: 'p' for site in sites}))


def add_inhibitor(model: Model, name: str, targets: List[str]):
    inh = Monomer(name, sites=['target'])
    Initial(inh(target=None), Parameter(f'{name}_0', 0.0), fixed=True)
    for target in targets:

        mono_name, site_conditions = site_states_from_string(target)
        koff = Parameter(f'{name}_{target}_koff', 0.1)
        kd = Parameter(f'{name}_{target}_kd', 1.0)
        kon = Expression(f'{name}_{target}_kon', kd*koff)
        if site_conditions:
            Rule(
                f'{name}_binds_{mono_name}',
                model.monomers[mono_name](inh=None, **site_conditions)
                +
                inh(target=None)
                >>
                model.monomers[mono_name](inh=1, **site_conditions)
                %
                inh(target=1),
                kon
            )
            Rule(
                f'{name}_unbinds_{mono_name}',
                model.monomers[mono_name](inh=1) % inh(target=1)
                >>
                model.monomers[mono_name](inh=None) + inh(target=None),
                koff
            )
        else:
            Rule(
                f'{name}_inhibits_{target}',
                model.monomers[target](inh=None) + inh(target=None)
                |
                model.monomers[target](inh=1) % inh(target=1),
                kon, koff
            )


def add_gf_bolus(model, name: str, created_monomers: List[str]):
    bolus = Monomer(f'{name}_ext')
    Initial(bolus(), Parameter(f'{name}_0', 0.0), fixed=True)
    for created_monomer in created_monomers:
        koff = Parameter(f'{name}_{created_monomer}_koff', 0.1)
        kd = Parameter(f'{name}_{created_monomer}_kd', 1.0)
        kon = Expression(f'{name}_{created_monomer}_kon', kd * koff)
        Rule(
            f'{name}_ext_to_{created_monomer}',
            bolus() | model.monomers[created_monomer](inh=None),
            kon, koff
        )








