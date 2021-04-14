import re
import itertools as itt
import sympy as sp

from typing import Iterable, Optional, Dict, Tuple, List
from pysb import (
    Monomer, Expression, Parameter, Rule, Model, Observable, Initial
)
import pysb.bng


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
                          asite_states: Optional[Iterable[str]] = None,
                          with_basal_activation: Optional[bool] = True):
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

    #kdeg = Parameter(f'{m_name}_degradation_kdeg', 1.0)
    t = Parameter(f'{m_name}_eq', 100.0)
    t0 = Expression(f'{m_name}_init', t * get_autoencoder_modulator(t))
    #ksyn = Expression(f'{m_name}_synthesis_ksyn', t*kdeg)
    #syn_rate = Expression(f'{m_name}_synthesis_rate',
    #                      ksyn * get_autoencoder_modulator(t))

    syn_prod = m(
        **{site:
           'u' if site in psites
           else 'gdp' if site in nsites
           else None if site == 'inh'
           else asite_states[0]
           for site in sites}
    )

    #Rule(f'synthesis_{m_name}', None >> syn_prod, syn_rate)
    #deg_rate = Expression(f'{m_name}_degradation_rate', kdeg)
    #Rule(f'degradation_{m_name}', m() >> None, deg_rate)
    #t_ss = Expression(f'{m_name}_ss', syn_rate/deg_rate)
    Initial(syn_prod, t0)

    if not with_basal_activation:
        return m

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
                for label in labels[1:]
            ]
            rates = [
                Expression(f'{m_name}_{label}_{site}_base_rate',
                           kcat * get_autoencoder_modulator(kcat))
                for kcat, label in zip(kcats, labels[1:])
            ]

            #Rule(F'{m_name}_{site}_base',
            #     m(**{site: rstate}) | m(**{site: fstate}), *rates)
            Rule(F'{m_name}_{site}_base',
                 m(**{site: fstate}) >> m(**{site: rstate}), *rates)

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

    sites = [s for s in site.split('_')]

    for s in sites:
        if s not in mono.site_states \
                or len(mono.site_states[s]) != len(valid_states) \
                or any(state not in mono.site_states[s]
                       for state in valid_states):
            raise ValueError(f'{s} is not a valid target for '
                             f'{activation_type}.')

    if activation_type == 'phosphorylation':
        forward = 'phosphorylation_' + site
        backward = 'dephosphorylation_' + site
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
    inh = Parameter(f'{name}_0', 0.0)
    kd = Parameter(f'{name}_kd', 0.0)
    affinities = {
        target: Expression(
            f'inh_{target}',
            Observable(f'target_{target}', model.monomers[target])/kd,
            _export=False
        )
        for target in targets
    }

    for expr in model.expressions:
        if expr.name.startswith('inh_'):
            continue
        target = next((
            next(
                mp.monomer.name
                for cp in s.reaction_pattern.complex_patterns
                for mp in cp.monomer_patterns
                if mp.monomer.name in targets
            )
            for s in expr.expr.free_symbols
            if isinstance(s, Observable) and any(
                mp.monomer.name in targets
                for cp in s.reaction_pattern.complex_patterns
                for mp in cp.monomer_patterns
            )
        ), None)
        if target is None:
            continue
        expr.expr *= 1/(1 + inh * affinities[target])

    model.expressions = pysb.ComponentSet(list(affinities.values()) +
                                          list(model.expressions))


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


def cleanup_unused(model):

    model.reset_equations()
    pysb.bng.generate_equations(model)

    observables = [
        obs.name for obs in model.expressions
        if obs.name.endswith('_obs')
    ]

    dynamic_eq = sp.Matrix(model.odes)

    expression_dynamic_symbols = set()
    for sym in dynamic_eq.free_symbols:
        if not isinstance(sym, Expression):
            continue
        if sym.name in model.expressions.keys():
            expression_dynamic_symbols |= model.expressions[
                sym.name
            ].expand_expr().free_symbols

    initial_eq = sp.Matrix([
        initial.value.expand_expr()
        if isinstance(initial.value, Expression) else initial.value
        for initial in model.initials
    ])

    observable_eq = sp.Matrix([
        expression.expand_expr()
        for expression in model.expressions
        if expression.name in observables
    ])

    free_symbols = list(
        dynamic_eq.free_symbols | initial_eq.free_symbols |
        observable_eq.free_symbols | expression_dynamic_symbols
    )

    unused_pars = set(
        par
        for par in model.parameters
        if par not in free_symbols and sp.Symbol(par.name) not in free_symbols
    )

    rule_reaction_count = {
        rule.name: 0
        for rule in model.rules
    }

    for reaction in model.reactions:
        for rule in reaction['rule']:
            rule_reaction_count[rule] += 1

    model.parameters = pysb.ComponentSet([
        par for par in model.parameters
        if par not in unused_pars
    ])

    model.expressions = pysb.ComponentSet([
        expr for expr in model.expressions
        if len(expr.expand_expr().free_symbols.intersection(unused_pars)) == 0
        and not expr.name.startswith('_')
    ])

    model.rules = pysb.ComponentSet([
        rule for rule in model.rules
        if rule_reaction_count[rule.name] > 0
    ])

    # model.observables = [
    #     obs for obs in model.observables
    #     if len(obs.coefficients) > 0
    # ]

    model.reset_equations()







