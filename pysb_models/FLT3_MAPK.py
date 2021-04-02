# exported from PySB model 'FLT3_MAPK'

from pysb import Model, Monomer, Parameter, Expression, Compartment, Rule, Observable, Initial, MatchOnce, Annotation, MultiState, Tag, ANY, WILD

Model()

Monomer('FL', ['inh'])
Monomer('FLT3', ['Y843', 'inh'], {'Y843': ['u', 'p']})
Monomer('HRAS', ['N', 'inh'], {'N': ['gdp', 'gtp']})
Monomer('KRAS', ['N', 'inh'], {'N': ['gdp', 'gtp']})
Monomer('NRAS', ['N', 'inh'], {'N': ['gdp', 'gtp']})
Monomer('RAF1', ['S338', 'inh'], {'S338': ['u', 'p']})
Monomer('BRAF', ['S447', 'inh'], {'S447': ['u', 'p']})
Monomer('MAP2K1', ['S218', 'S222', 'inh'], {'S218': ['u', 'p'], 'S222': ['u', 'p']})
Monomer('MAP2K2', ['S226', 'S222', 'inh'], {'S226': ['u', 'p'], 'S222': ['u', 'p']})
Monomer('MAPK1', ['Y187', 'T185', 'inh'], {'Y187': ['u', 'p'], 'T185': ['u', 'p']})
Monomer('MAPK3', ['T202', 'Y204', 'inh'], {'T202': ['u', 'p'], 'Y204': ['u', 'p']})

Parameter('FL_eq', 100.0)
Parameter('FLT3_eq', 100.0)
Parameter('FLT3_dephosphorylation_Y843_base_kcat', 1.0)
Parameter('INPUT_FLT3_dephosphorylation_Y843_base_kcat', 0.0)
Parameter('FLT3_phosphorylation_Y843_FL_kcat', 1.0)
Parameter('INPUT_FLT3_phosphorylation_Y843_FL_kcat', 0.0)
Parameter('HRAS_eq', 100.0)
Parameter('HRAS_gdp_exchange_N_base_kcat', 1.0)
Parameter('INPUT_HRAS_gdp_exchange_N_base_kcat', 0.0)
Parameter('HRAS_gtp_exchange_N_FLT3__Y843_p_kcat', 1.0)
Parameter('INPUT_HRAS_gtp_exchange_N_FLT3__Y843_p_kcat', 0.0)
Parameter('KRAS_eq', 100.0)
Parameter('KRAS_gdp_exchange_N_base_kcat', 1.0)
Parameter('INPUT_KRAS_gdp_exchange_N_base_kcat', 0.0)
Parameter('KRAS_gtp_exchange_N_FLT3__Y843_p_kcat', 1.0)
Parameter('INPUT_KRAS_gtp_exchange_N_FLT3__Y843_p_kcat', 0.0)
Parameter('NRAS_eq', 100.0)
Parameter('NRAS_gdp_exchange_N_base_kcat', 1.0)
Parameter('INPUT_NRAS_gdp_exchange_N_base_kcat', 0.0)
Parameter('NRAS_gtp_exchange_N_FLT3__Y843_p_kcat', 1.0)
Parameter('INPUT_NRAS_gtp_exchange_N_FLT3__Y843_p_kcat', 0.0)
Parameter('RAF1_eq', 100.0)
Parameter('RAF1_dephosphorylation_S338_base_kcat', 1.0)
Parameter('INPUT_RAF1_dephosphorylation_S338_base_kcat', 0.0)
Parameter('BRAF_eq', 100.0)
Parameter('BRAF_dephosphorylation_S447_base_kcat', 1.0)
Parameter('INPUT_BRAF_dephosphorylation_S447_base_kcat', 0.0)
Parameter('MAP2K1_eq', 100.0)
Parameter('MAP2K1_dephosphorylation_S218_base_kcat', 1.0)
Parameter('INPUT_MAP2K1_dephosphorylation_S218_base_kcat', 0.0)
Parameter('MAP2K1_dephosphorylation_S222_base_kcat', 1.0)
Parameter('INPUT_MAP2K1_dephosphorylation_S222_base_kcat', 0.0)
Parameter('MAP2K2_eq', 100.0)
Parameter('MAP2K2_dephosphorylation_S226_base_kcat', 1.0)
Parameter('INPUT_MAP2K2_dephosphorylation_S226_base_kcat', 0.0)
Parameter('MAP2K2_dephosphorylation_S222_base_kcat', 1.0)
Parameter('INPUT_MAP2K2_dephosphorylation_S222_base_kcat', 0.0)
Parameter('MAPK1_eq', 100.0)
Parameter('MAPK1_dephosphorylation_Y187_base_kcat', 1.0)
Parameter('INPUT_MAPK1_dephosphorylation_Y187_base_kcat', 0.0)
Parameter('MAPK1_dephosphorylation_T185_base_kcat', 1.0)
Parameter('INPUT_MAPK1_dephosphorylation_T185_base_kcat', 0.0)
Parameter('MAPK3_eq', 100.0)
Parameter('MAPK3_dephosphorylation_T202_base_kcat', 1.0)
Parameter('INPUT_MAPK3_dephosphorylation_T202_base_kcat', 0.0)
Parameter('MAPK3_dephosphorylation_Y204_base_kcat', 1.0)
Parameter('INPUT_MAPK3_dephosphorylation_Y204_base_kcat', 0.0)
Parameter('RAF1_phosphorylation_S338_KRAS__N_gtp_kcat', 1.0)
Parameter('INPUT_RAF1_phosphorylation_S338_KRAS__N_gtp_kcat', 0.0)
Parameter('RAF1_phosphorylation_S338_HRAS__N_gtp_kcat', 1.0)
Parameter('INPUT_RAF1_phosphorylation_S338_HRAS__N_gtp_kcat', 0.0)
Parameter('RAF1_phosphorylation_S338_NRAS__N_gtp_kcat', 1.0)
Parameter('INPUT_RAF1_phosphorylation_S338_NRAS__N_gtp_kcat', 0.0)
Parameter('BRAF_phosphorylation_S447_KRAS__N_gtp_kcat', 1.0)
Parameter('INPUT_BRAF_phosphorylation_S447_KRAS__N_gtp_kcat', 0.0)
Parameter('BRAF_phosphorylation_S447_HRAS__N_gtp_kcat', 1.0)
Parameter('INPUT_BRAF_phosphorylation_S447_HRAS__N_gtp_kcat', 0.0)
Parameter('BRAF_phosphorylation_S447_NRAS__N_gtp_kcat', 1.0)
Parameter('INPUT_BRAF_phosphorylation_S447_NRAS__N_gtp_kcat', 0.0)
Parameter('MAP2K1_phosphorylation_S222_RAF1__S338_p_kcat', 1.0)
Parameter('INPUT_MAP2K1_phosphorylation_S222_RAF1__S338_p_kcat', 0.0)
Parameter('MAP2K1_phosphorylation_S222_BRAF__S447_p_kcat', 1.0)
Parameter('INPUT_MAP2K1_phosphorylation_S222_BRAF__S447_p_kcat', 0.0)
Parameter('MAP2K2_phosphorylation_S226_RAF1__S338_p_kcat', 1.0)
Parameter('INPUT_MAP2K2_phosphorylation_S226_RAF1__S338_p_kcat', 0.0)
Parameter('MAP2K2_phosphorylation_S226_BRAF__S447_p_kcat', 1.0)
Parameter('INPUT_MAP2K2_phosphorylation_S226_BRAF__S447_p_kcat', 0.0)
Parameter('MAPK1_phosphorylation_Y187_MAP2K1__S218_p__S222_p_kcat', 1.0)
Parameter('INPUT_MAPK1_phosphorylation_Y187_MAP2K1__S218_p__S222_p_kcat', 0.0)
Parameter('MAPK1_phosphorylation_Y187_MAP2K2__S222_p__S226_p_kcat', 1.0)
Parameter('INPUT_MAPK1_phosphorylation_Y187_MAP2K2__S222_p__S226_p_kcat', 0.0)
Parameter('MAPK3_phosphorylation_Y204_MAP2K1__S218_p__S222_p_kcat', 1.0)
Parameter('INPUT_MAPK3_phosphorylation_Y204_MAP2K1__S218_p__S222_p_kcat', 0.0)
Parameter('MAPK3_phosphorylation_Y204_MAP2K2__S222_p__S226_p_kcat', 1.0)
Parameter('INPUT_MAPK3_phosphorylation_Y204_MAP2K2__S222_p__S226_p_kcat', 0.0)

Expression('FLT3_dephosphorylation_Y843_base_rate', FLT3_dephosphorylation_Y843_base_kcat*INPUT_FLT3_dephosphorylation_Y843_base_kcat)
Expression('HRAS_gdp_exchange_N_base_rate', HRAS_gdp_exchange_N_base_kcat*INPUT_HRAS_gdp_exchange_N_base_kcat)
Expression('KRAS_gdp_exchange_N_base_rate', INPUT_KRAS_gdp_exchange_N_base_kcat*KRAS_gdp_exchange_N_base_kcat)
Expression('NRAS_gdp_exchange_N_base_rate', INPUT_NRAS_gdp_exchange_N_base_kcat*NRAS_gdp_exchange_N_base_kcat)
Expression('RAF1_dephosphorylation_S338_base_rate', INPUT_RAF1_dephosphorylation_S338_base_kcat*RAF1_dephosphorylation_S338_base_kcat)
Expression('BRAF_dephosphorylation_S447_base_rate', BRAF_dephosphorylation_S447_base_kcat*INPUT_BRAF_dephosphorylation_S447_base_kcat)
Expression('MAP2K1_dephosphorylation_S218_base_rate', INPUT_MAP2K1_dephosphorylation_S218_base_kcat*MAP2K1_dephosphorylation_S218_base_kcat)
Expression('MAP2K1_dephosphorylation_S222_base_rate', INPUT_MAP2K1_dephosphorylation_S222_base_kcat*MAP2K1_dephosphorylation_S222_base_kcat)
Expression('MAP2K2_dephosphorylation_S226_base_rate', INPUT_MAP2K2_dephosphorylation_S226_base_kcat*MAP2K2_dephosphorylation_S226_base_kcat)
Expression('MAP2K2_dephosphorylation_S222_base_rate', INPUT_MAP2K2_dephosphorylation_S222_base_kcat*MAP2K2_dephosphorylation_S222_base_kcat)
Expression('MAPK1_dephosphorylation_Y187_base_rate', INPUT_MAPK1_dephosphorylation_Y187_base_kcat*MAPK1_dephosphorylation_Y187_base_kcat)
Expression('MAPK1_dephosphorylation_T185_base_rate', INPUT_MAPK1_dephosphorylation_T185_base_kcat*MAPK1_dephosphorylation_T185_base_kcat)
Expression('MAPK3_dephosphorylation_T202_base_rate', INPUT_MAPK3_dephosphorylation_T202_base_kcat*MAPK3_dephosphorylation_T202_base_kcat)
Expression('MAPK3_dephosphorylation_Y204_base_rate', INPUT_MAPK3_dephosphorylation_Y204_base_kcat*MAPK3_dephosphorylation_Y204_base_kcat)

Observable('FL_obs', FL(inh=None))
Observable('FLT3__Y843_p_obs', FLT3(Y843='p', inh=None))
Observable('KRAS__N_gtp_obs', KRAS(N='gtp', inh=None))
Observable('HRAS__N_gtp_obs', HRAS(N='gtp', inh=None))
Observable('NRAS__N_gtp_obs', NRAS(N='gtp', inh=None))
Observable('RAF1__S338_p_obs', RAF1(S338='p', inh=None))
Observable('BRAF__S447_p_obs', BRAF(S447='p', inh=None))
Observable('MAP2K1__S218_p__S222_p_obs', MAP2K1(S218='p', S222='p', inh=None))
Observable('MAP2K2__S222_p__S226_p_obs', MAP2K2(S226='p', S222='p', inh=None))
Observable('tFL', FL())
Observable('tFLT3', FLT3())
Observable('pFLT3_Y843', FLT3(Y843='p'))
Observable('tHRAS', HRAS())
Observable('tKRAS', KRAS())
Observable('tNRAS', NRAS())
Observable('tRAF1', RAF1())
Observable('pRAF1_S338', RAF1(S338='p'))
Observable('tBRAF', BRAF())
Observable('pBRAF_S447', BRAF(S447='p'))
Observable('tMAP2K1', MAP2K1())
Observable('pMAP2K1_S218', MAP2K1(S218='p'))
Observable('pMAP2K1_S222', MAP2K1(S222='p'))
Observable('pMAP2K1_S218_S222', MAP2K1(S218='p', S222='p'))
Observable('tMAP2K2', MAP2K2())
Observable('pMAP2K2_S226', MAP2K2(S226='p'))
Observable('pMAP2K2_S222', MAP2K2(S222='p'))
Observable('pMAP2K2_S222_S226', MAP2K2(S226='p', S222='p'))
Observable('tMAPK1', MAPK1())
Observable('pMAPK1_Y187', MAPK1(Y187='p'))
Observable('pMAPK1_T185', MAPK1(T185='p'))
Observable('pMAPK1_T185_Y187', MAPK1(Y187='p', T185='p'))
Observable('tMAPK3', MAPK3())
Observable('pMAPK3_T202', MAPK3(T202='p'))
Observable('pMAPK3_Y204', MAPK3(Y204='p'))
Observable('pMAPK3_T202_Y204', MAPK3(T202='p', Y204='p'))

Expression('FLT3_phosphorylation_Y843_FL_rate', FL_obs*FLT3_phosphorylation_Y843_FL_kcat*INPUT_FLT3_phosphorylation_Y843_FL_kcat)
Expression('HRAS_gtp_exchange_N_FLT3__Y843_p_rate', FLT3__Y843_p_obs*HRAS_gtp_exchange_N_FLT3__Y843_p_kcat*INPUT_HRAS_gtp_exchange_N_FLT3__Y843_p_kcat)
Expression('KRAS_gtp_exchange_N_FLT3__Y843_p_rate', FLT3__Y843_p_obs*INPUT_KRAS_gtp_exchange_N_FLT3__Y843_p_kcat*KRAS_gtp_exchange_N_FLT3__Y843_p_kcat)
Expression('NRAS_gtp_exchange_N_FLT3__Y843_p_rate', FLT3__Y843_p_obs*INPUT_NRAS_gtp_exchange_N_FLT3__Y843_p_kcat*NRAS_gtp_exchange_N_FLT3__Y843_p_kcat)
Expression('RAF1_phosphorylation_S338_KRAS__N_gtp_rate', KRAS__N_gtp_obs*INPUT_RAF1_phosphorylation_S338_KRAS__N_gtp_kcat*RAF1_phosphorylation_S338_KRAS__N_gtp_kcat)
Expression('RAF1_phosphorylation_S338_HRAS__N_gtp_rate', HRAS__N_gtp_obs*INPUT_RAF1_phosphorylation_S338_HRAS__N_gtp_kcat*RAF1_phosphorylation_S338_HRAS__N_gtp_kcat)
Expression('RAF1_phosphorylation_S338_NRAS__N_gtp_rate', NRAS__N_gtp_obs*INPUT_RAF1_phosphorylation_S338_NRAS__N_gtp_kcat*RAF1_phosphorylation_S338_NRAS__N_gtp_kcat)
Expression('BRAF_phosphorylation_S447_KRAS__N_gtp_rate', KRAS__N_gtp_obs*BRAF_phosphorylation_S447_KRAS__N_gtp_kcat*INPUT_BRAF_phosphorylation_S447_KRAS__N_gtp_kcat)
Expression('BRAF_phosphorylation_S447_HRAS__N_gtp_rate', HRAS__N_gtp_obs*BRAF_phosphorylation_S447_HRAS__N_gtp_kcat*INPUT_BRAF_phosphorylation_S447_HRAS__N_gtp_kcat)
Expression('BRAF_phosphorylation_S447_NRAS__N_gtp_rate', NRAS__N_gtp_obs*BRAF_phosphorylation_S447_NRAS__N_gtp_kcat*INPUT_BRAF_phosphorylation_S447_NRAS__N_gtp_kcat)
Expression('MAP2K1_phosphorylation_S222_RAF1__S338_p_rate', RAF1__S338_p_obs*INPUT_MAP2K1_phosphorylation_S222_RAF1__S338_p_kcat*MAP2K1_phosphorylation_S222_RAF1__S338_p_kcat)
Expression('MAP2K1_phosphorylation_S222_BRAF__S447_p_rate', BRAF__S447_p_obs*INPUT_MAP2K1_phosphorylation_S222_BRAF__S447_p_kcat*MAP2K1_phosphorylation_S222_BRAF__S447_p_kcat)
Expression('MAP2K2_phosphorylation_S226_RAF1__S338_p_rate', RAF1__S338_p_obs*INPUT_MAP2K2_phosphorylation_S226_RAF1__S338_p_kcat*MAP2K2_phosphorylation_S226_RAF1__S338_p_kcat)
Expression('MAP2K2_phosphorylation_S226_BRAF__S447_p_rate', BRAF__S447_p_obs*INPUT_MAP2K2_phosphorylation_S226_BRAF__S447_p_kcat*MAP2K2_phosphorylation_S226_BRAF__S447_p_kcat)
Expression('MAPK1_phosphorylation_Y187_MAP2K1__S218_p__S222_p_rate', MAP2K1__S218_p__S222_p_obs*INPUT_MAPK1_phosphorylation_Y187_MAP2K1__S218_p__S222_p_kcat*MAPK1_phosphorylation_Y187_MAP2K1__S218_p__S222_p_kcat)
Expression('MAPK1_phosphorylation_Y187_MAP2K2__S222_p__S226_p_rate', MAP2K2__S222_p__S226_p_obs*INPUT_MAPK1_phosphorylation_Y187_MAP2K2__S222_p__S226_p_kcat*MAPK1_phosphorylation_Y187_MAP2K2__S222_p__S226_p_kcat)
Expression('MAPK3_phosphorylation_Y204_MAP2K1__S218_p__S222_p_rate', MAP2K1__S218_p__S222_p_obs*INPUT_MAPK3_phosphorylation_Y204_MAP2K1__S218_p__S222_p_kcat*MAPK3_phosphorylation_Y204_MAP2K1__S218_p__S222_p_kcat)
Expression('MAPK3_phosphorylation_Y204_MAP2K2__S222_p__S226_p_rate', MAP2K2__S222_p__S226_p_obs*INPUT_MAPK3_phosphorylation_Y204_MAP2K2__S222_p__S226_p_kcat*MAPK3_phosphorylation_Y204_MAP2K2__S222_p__S226_p_kcat)

Rule('FLT3_Y843_base', FLT3(Y843='p') >> FLT3(Y843='u'), FLT3_dephosphorylation_Y843_base_rate)
Rule('FLT3_phosphorylation_Y843_FL', FLT3(Y843='u') >> FLT3(Y843='p'), FLT3_phosphorylation_Y843_FL_rate)
Rule('HRAS_N_base', HRAS(N='gtp') >> HRAS(N='gdp'), HRAS_gdp_exchange_N_base_rate)
Rule('HRAS_gtp_exchange_N_FLT3__Y843_p', HRAS(N='gdp') >> HRAS(N='gtp'), HRAS_gtp_exchange_N_FLT3__Y843_p_rate)
Rule('KRAS_N_base', KRAS(N='gtp') >> KRAS(N='gdp'), KRAS_gdp_exchange_N_base_rate)
Rule('KRAS_gtp_exchange_N_FLT3__Y843_p', KRAS(N='gdp') >> KRAS(N='gtp'), KRAS_gtp_exchange_N_FLT3__Y843_p_rate)
Rule('NRAS_N_base', NRAS(N='gtp') >> NRAS(N='gdp'), NRAS_gdp_exchange_N_base_rate)
Rule('NRAS_gtp_exchange_N_FLT3__Y843_p', NRAS(N='gdp') >> NRAS(N='gtp'), NRAS_gtp_exchange_N_FLT3__Y843_p_rate)
Rule('RAF1_S338_base', RAF1(S338='p') >> RAF1(S338='u'), RAF1_dephosphorylation_S338_base_rate)
Rule('BRAF_S447_base', BRAF(S447='p') >> BRAF(S447='u'), BRAF_dephosphorylation_S447_base_rate)
Rule('MAP2K1_S218_base', MAP2K1(S218='p') >> MAP2K1(S218='u'), MAP2K1_dephosphorylation_S218_base_rate)
Rule('MAP2K1_S222_base', MAP2K1(S222='p') >> MAP2K1(S222='u'), MAP2K1_dephosphorylation_S222_base_rate)
Rule('MAP2K2_S226_base', MAP2K2(S226='p') >> MAP2K2(S226='u'), MAP2K2_dephosphorylation_S226_base_rate)
Rule('MAP2K2_S222_base', MAP2K2(S222='p') >> MAP2K2(S222='u'), MAP2K2_dephosphorylation_S222_base_rate)
Rule('MAPK1_Y187_base', MAPK1(Y187='p') >> MAPK1(Y187='u'), MAPK1_dephosphorylation_Y187_base_rate)
Rule('MAPK1_T185_base', MAPK1(T185='p') >> MAPK1(T185='u'), MAPK1_dephosphorylation_T185_base_rate)
Rule('MAPK3_T202_base', MAPK3(T202='p') >> MAPK3(T202='u'), MAPK3_dephosphorylation_T202_base_rate)
Rule('MAPK3_Y204_base', MAPK3(Y204='p') >> MAPK3(Y204='u'), MAPK3_dephosphorylation_Y204_base_rate)
Rule('RAF1_phosphorylation_S338_KRAS__N_gtp', RAF1(S338='u') >> RAF1(S338='p'), RAF1_phosphorylation_S338_KRAS__N_gtp_rate)
Rule('RAF1_phosphorylation_S338_HRAS__N_gtp', RAF1(S338='u') >> RAF1(S338='p'), RAF1_phosphorylation_S338_HRAS__N_gtp_rate)
Rule('RAF1_phosphorylation_S338_NRAS__N_gtp', RAF1(S338='u') >> RAF1(S338='p'), RAF1_phosphorylation_S338_NRAS__N_gtp_rate)
Rule('BRAF_phosphorylation_S447_KRAS__N_gtp', BRAF(S447='u') >> BRAF(S447='p'), BRAF_phosphorylation_S447_KRAS__N_gtp_rate)
Rule('BRAF_phosphorylation_S447_HRAS__N_gtp', BRAF(S447='u') >> BRAF(S447='p'), BRAF_phosphorylation_S447_HRAS__N_gtp_rate)
Rule('BRAF_phosphorylation_S447_NRAS__N_gtp', BRAF(S447='u') >> BRAF(S447='p'), BRAF_phosphorylation_S447_NRAS__N_gtp_rate)
Rule('MAP2K1_phosphorylation_S222_RAF1__S338_p', MAP2K1(S218='u', S222='u') >> MAP2K1(S218='p', S222='p'), MAP2K1_phosphorylation_S222_RAF1__S338_p_rate)
Rule('MAP2K1_phosphorylation_S222_BRAF__S447_p', MAP2K1(S218='u', S222='u') >> MAP2K1(S218='p', S222='p'), MAP2K1_phosphorylation_S222_BRAF__S447_p_rate)
Rule('MAP2K2_phosphorylation_S226_RAF1__S338_p', MAP2K2(S226='u', S222='u') >> MAP2K2(S226='p', S222='p'), MAP2K2_phosphorylation_S226_RAF1__S338_p_rate)
Rule('MAP2K2_phosphorylation_S226_BRAF__S447_p', MAP2K2(S226='u', S222='u') >> MAP2K2(S226='p', S222='p'), MAP2K2_phosphorylation_S226_BRAF__S447_p_rate)
Rule('MAPK1_phosphorylation_Y187_MAP2K1__S218_p__S222_p', MAPK1(Y187='u', T185='u') >> MAPK1(Y187='p', T185='p'), MAPK1_phosphorylation_Y187_MAP2K1__S218_p__S222_p_rate)
Rule('MAPK1_phosphorylation_Y187_MAP2K2__S222_p__S226_p', MAPK1(Y187='u', T185='u') >> MAPK1(Y187='p', T185='p'), MAPK1_phosphorylation_Y187_MAP2K2__S222_p__S226_p_rate)
Rule('MAPK3_phosphorylation_Y204_MAP2K1__S218_p__S222_p', MAPK3(T202='u', Y204='u') >> MAPK3(T202='p', Y204='p'), MAPK3_phosphorylation_Y204_MAP2K1__S218_p__S222_p_rate)
Rule('MAPK3_phosphorylation_Y204_MAP2K2__S222_p__S226_p', MAPK3(T202='u', Y204='u') >> MAPK3(T202='p', Y204='p'), MAPK3_phosphorylation_Y204_MAP2K2__S222_p__S226_p_rate)

Initial(FL(inh=None), FL_eq)
Initial(FLT3(Y843='u', inh=None), FLT3_eq)
Initial(HRAS(N='gdp', inh=None), HRAS_eq)
Initial(KRAS(N='gdp', inh=None), KRAS_eq)
Initial(NRAS(N='gdp', inh=None), NRAS_eq)
Initial(RAF1(S338='u', inh=None), RAF1_eq)
Initial(BRAF(S447='u', inh=None), BRAF_eq)
Initial(MAP2K1(S218='u', S222='u', inh=None), MAP2K1_eq)
Initial(MAP2K2(S226='u', S222='u', inh=None), MAP2K2_eq)
Initial(MAPK1(Y187='u', T185='u', inh=None), MAPK1_eq)
Initial(MAPK3(T202='u', Y204='u', inh=None), MAPK3_eq)

