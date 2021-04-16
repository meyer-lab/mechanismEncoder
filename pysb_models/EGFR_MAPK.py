# exported from PySB model 'EGFR_MAPK'

from pysb import Model, Monomer, Parameter, Expression, Compartment, Rule, Observable, Initial, MatchOnce, Annotation, MultiState, Tag, ANY, WILD

Model()

Monomer('EGF', ['inh'])
Monomer('EGF_ext')
Monomer('EGFR', ['Y1173', 'inh'], {'Y1173': ['u', 'p']})
Monomer('ERBB2', ['Y1248', 'inh'], {'Y1248': ['u', 'p']})
Monomer('RAF1', ['S338', 'inh'], {'S338': ['u', 'p']})
Monomer('BRAF', ['S445', 'inh'], {'S445': ['u', 'p']})
Monomer('MAP2K1', ['S222', 'S218', 'inh'], {'S222': ['u', 'p'], 'S218': ['u', 'p']})
Monomer('MAP2K2', ['S222', 'S226', 'inh'], {'S222': ['u', 'p'], 'S226': ['u', 'p']})
Monomer('MAPK1', ['Y187', 'T185', 'inh'], {'Y187': ['u', 'p'], 'T185': ['u', 'p']})
Monomer('MAPK3', ['T202', 'Y204', 'inh'], {'T202': ['u', 'p'], 'Y204': ['u', 'p']})

Parameter('EGF_eq', 100.0)
Parameter('INPUT_EGF_eq', 0.0)
Parameter('EGF_0', 0.0)
Parameter('EGF_EGF_koff', 0.1)
Parameter('EGF_EGF_kd', 1.0)
Parameter('EGFR_eq', 100.0)
Parameter('INPUT_EGFR_eq', 0.0)
Parameter('EGFR_dephosphorylation_Y1173_base_kcat', 1.0)
Parameter('INPUT_EGFR_dephosphorylation_Y1173_base_kcat', 0.0)
Parameter('ERBB2_eq', 100.0)
Parameter('INPUT_ERBB2_eq', 0.0)
Parameter('ERBB2_dephosphorylation_Y1248_base_kcat', 1.0)
Parameter('INPUT_ERBB2_dephosphorylation_Y1248_base_kcat', 0.0)
Parameter('EGFR_phosphorylation_Y1173_Y1173_EGF_kcat', 1.0)
Parameter('INPUT_EGFR_phosphorylation_Y1173_Y1173_EGF_kcat', 0.0)
Parameter('ERBB2_phosphorylation_Y1248_Y1248_EGFR__Y1173_p_kcat', 1.0)
Parameter('INPUT_ERBB2_phosphorylation_Y1248_Y1248_EGFR__Y1173_p_kcat', 0.0)
Parameter('RAF1_eq', 100.0)
Parameter('INPUT_RAF1_eq', 0.0)
Parameter('RAF1_dephosphorylation_S338_base_kcat', 1.0)
Parameter('INPUT_RAF1_dephosphorylation_S338_base_kcat', 0.0)
Parameter('BRAF_eq', 100.0)
Parameter('INPUT_BRAF_eq', 0.0)
Parameter('BRAF_dephosphorylation_S445_base_kcat', 1.0)
Parameter('INPUT_BRAF_dephosphorylation_S445_base_kcat', 0.0)
Parameter('MAP2K1_eq', 100.0)
Parameter('INPUT_MAP2K1_eq', 0.0)
Parameter('MAP2K1_dephosphorylation_S222_base_kcat', 1.0)
Parameter('INPUT_MAP2K1_dephosphorylation_S222_base_kcat', 0.0)
Parameter('MAP2K1_dephosphorylation_S218_base_kcat', 1.0)
Parameter('INPUT_MAP2K1_dephosphorylation_S218_base_kcat', 0.0)
Parameter('MAP2K2_eq', 100.0)
Parameter('INPUT_MAP2K2_eq', 0.0)
Parameter('MAP2K2_dephosphorylation_S222_base_kcat', 1.0)
Parameter('INPUT_MAP2K2_dephosphorylation_S222_base_kcat', 0.0)
Parameter('MAP2K2_dephosphorylation_S226_base_kcat', 1.0)
Parameter('INPUT_MAP2K2_dephosphorylation_S226_base_kcat', 0.0)
Parameter('MAPK1_eq', 100.0)
Parameter('INPUT_MAPK1_eq', 0.0)
Parameter('MAPK1_dephosphorylation_Y187_base_kcat', 1.0)
Parameter('INPUT_MAPK1_dephosphorylation_Y187_base_kcat', 0.0)
Parameter('MAPK1_dephosphorylation_T185_base_kcat', 1.0)
Parameter('INPUT_MAPK1_dephosphorylation_T185_base_kcat', 0.0)
Parameter('MAPK3_eq', 100.0)
Parameter('INPUT_MAPK3_eq', 0.0)
Parameter('MAPK3_dephosphorylation_T202_base_kcat', 1.0)
Parameter('INPUT_MAPK3_dephosphorylation_T202_base_kcat', 0.0)
Parameter('MAPK3_dephosphorylation_Y204_base_kcat', 1.0)
Parameter('INPUT_MAPK3_dephosphorylation_Y204_base_kcat', 0.0)
Parameter('RAF1_phosphorylation_S338_S338_EGFR__Y1173_p_kcat', 1.0)
Parameter('INPUT_RAF1_phosphorylation_S338_S338_EGFR__Y1173_p_kcat', 0.0)
Parameter('RAF1_phosphorylation_S338_S338_ERBB2__Y1248_p_kcat', 1.0)
Parameter('INPUT_RAF1_phosphorylation_S338_S338_ERBB2__Y1248_p_kcat', 0.0)
Parameter('BRAF_phosphorylation_S445_S445_EGFR__Y1173_p_kcat', 1.0)
Parameter('INPUT_BRAF_phosphorylation_S445_S445_EGFR__Y1173_p_kcat', 0.0)
Parameter('BRAF_phosphorylation_S445_S445_ERBB2__Y1248_p_kcat', 1.0)
Parameter('INPUT_BRAF_phosphorylation_S445_S445_ERBB2__Y1248_p_kcat', 0.0)
Parameter('MAP2K1_phosphorylation_S218_S222_S218_S222_RAF1__S338_p_kcat', 1.0)
Parameter('INPUT_MAP2K1_phosphorylation_S218_S222_S218_S222_RAF1__S338_p_kcat', 0.0)
Parameter('MAP2K1_phosphorylation_S218_S222_S218_S222_BRAF__S445_p_kcat', 1.0)
Parameter('INPUT_MAP2K1_phosphorylation_S218_S222_S218_S222_BRAF__S445_p_kcat', 0.0)
Parameter('MAP2K2_phosphorylation_S222_S226_S222_S226_RAF1__S338_p_kcat', 1.0)
Parameter('INPUT_MAP2K2_phosphorylation_S222_S226_S222_S226_RAF1__S338_p_kcat', 0.0)
Parameter('MAP2K2_phosphorylation_S222_S226_S222_S226_BRAF__S445_p_kcat', 1.0)
Parameter('INPUT_MAP2K2_phosphorylation_S222_S226_S222_S226_BRAF__S445_p_kcat', 0.0)
Parameter('MAPK1_phosphorylation_T185_Y187_T185_Y187_MAP2K1__S218_p__S222_p_kcat', 1.0)
Parameter('INPUT_MAPK1_phosphorylation_T185_Y187_T185_Y187_MAP2K1__S218_p__S222_p_kcat', 0.0)
Parameter('MAPK1_phosphorylation_T185_Y187_T185_Y187_MAP2K2__S222_p__S226_p_kcat', 1.0)
Parameter('INPUT_MAPK1_phosphorylation_T185_Y187_T185_Y187_MAP2K2__S222_p__S226_p_kcat', 0.0)
Parameter('MAPK3_phosphorylation_T202_Y204_T202_Y204_MAP2K1__S218_p__S222_p_kcat', 1.0)
Parameter('INPUT_MAPK3_phosphorylation_T202_Y204_T202_Y204_MAP2K1__S218_p__S222_p_kcat', 0.0)
Parameter('MAPK3_phosphorylation_T202_Y204_T202_Y204_MAP2K2__S222_p__S226_p_kcat', 1.0)
Parameter('INPUT_MAPK3_phosphorylation_T202_Y204_T202_Y204_MAP2K2__S222_p__S226_p_kcat', 0.0)
Parameter('iEGFR_0', 0.0)
Parameter('iEGFR_kd', 0.0)
Parameter('iMEK_0', 0.0)
Parameter('iMEK_kd', 0.0)

Expression('EGF_init', EGF_eq*INPUT_EGF_eq)
Expression('EGF_EGF_kon', EGF_EGF_kd*EGF_EGF_koff)
Expression('EGFR_init', EGFR_eq*INPUT_EGFR_eq)
Expression('EGFR_dephosphorylation_Y1173_base_rate', EGFR_dephosphorylation_Y1173_base_kcat*INPUT_EGFR_dephosphorylation_Y1173_base_kcat)
Expression('ERBB2_init', ERBB2_eq*INPUT_ERBB2_eq)
Expression('ERBB2_dephosphorylation_Y1248_base_rate', ERBB2_dephosphorylation_Y1248_base_kcat*INPUT_ERBB2_dephosphorylation_Y1248_base_kcat)
Expression('RAF1_init', INPUT_RAF1_eq*RAF1_eq)
Expression('RAF1_dephosphorylation_S338_base_rate', INPUT_RAF1_dephosphorylation_S338_base_kcat*RAF1_dephosphorylation_S338_base_kcat)
Expression('BRAF_init', BRAF_eq*INPUT_BRAF_eq)
Expression('BRAF_dephosphorylation_S445_base_rate', BRAF_dephosphorylation_S445_base_kcat*INPUT_BRAF_dephosphorylation_S445_base_kcat)
Expression('MAP2K1_init', INPUT_MAP2K1_eq*MAP2K1_eq)
Expression('MAP2K1_dephosphorylation_S222_base_rate', INPUT_MAP2K1_dephosphorylation_S222_base_kcat*MAP2K1_dephosphorylation_S222_base_kcat)
Expression('MAP2K1_dephosphorylation_S218_base_rate', INPUT_MAP2K1_dephosphorylation_S218_base_kcat*MAP2K1_dephosphorylation_S218_base_kcat)
Expression('MAP2K2_init', INPUT_MAP2K2_eq*MAP2K2_eq)
Expression('MAP2K2_dephosphorylation_S222_base_rate', INPUT_MAP2K2_dephosphorylation_S222_base_kcat*MAP2K2_dephosphorylation_S222_base_kcat)
Expression('MAP2K2_dephosphorylation_S226_base_rate', INPUT_MAP2K2_dephosphorylation_S226_base_kcat*MAP2K2_dephosphorylation_S226_base_kcat)
Expression('MAPK1_init', INPUT_MAPK1_eq*MAPK1_eq)
Expression('MAPK1_dephosphorylation_Y187_base_rate', INPUT_MAPK1_dephosphorylation_Y187_base_kcat*MAPK1_dephosphorylation_Y187_base_kcat)
Expression('MAPK1_dephosphorylation_T185_base_rate', INPUT_MAPK1_dephosphorylation_T185_base_kcat*MAPK1_dephosphorylation_T185_base_kcat)
Expression('MAPK3_init', INPUT_MAPK3_eq*MAPK3_eq)
Expression('MAPK3_dephosphorylation_T202_base_rate', INPUT_MAPK3_dephosphorylation_T202_base_kcat*MAPK3_dephosphorylation_T202_base_kcat)
Expression('MAPK3_dephosphorylation_Y204_base_rate', INPUT_MAPK3_dephosphorylation_Y204_base_kcat*MAPK3_dephosphorylation_Y204_base_kcat)

Observable('EGF_obs', EGF(inh=None))
Observable('EGFR__Y1173_p_obs', EGFR(Y1173='p', inh=None))
Observable('ERBB2__Y1248_p_obs', ERBB2(Y1248='p', inh=None))
Observable('RAF1__S338_p_obs', RAF1(S338='p', inh=None))
Observable('BRAF__S445_p_obs', BRAF(S445='p', inh=None))
Observable('MAP2K1__S218_p__S222_p_obs', MAP2K1(S222='p', S218='p', inh=None))
Observable('MAP2K2__S222_p__S226_p_obs', MAP2K2(S222='p', S226='p', inh=None))
Observable('ERK_T202_Y204', MAPK1(Y187='p', T185='p') + MAPK3(T202='p', Y204='p'))
Observable('MEK_S221', MAP2K1(S222='p') + MAP2K2(S226='p'))
Observable('target_EGFR', EGFR())
Observable('target_MAP2K1', MAP2K1())
Observable('target_MAP2K2', MAP2K2())
Observable('tEGF', EGF())
Observable('tEGF_ext', EGF_ext())
Observable('tEGFR', EGFR())
Observable('pEGFR_Y1173', EGFR(Y1173='p'))
Observable('tERBB2', ERBB2())
Observable('pERBB2_Y1248', ERBB2(Y1248='p'))
Observable('tRAF1', RAF1())
Observable('pRAF1_S338', RAF1(S338='p'))
Observable('tBRAF', BRAF())
Observable('pBRAF_S445', BRAF(S445='p'))
Observable('tMAP2K1', MAP2K1())
Observable('pMAP2K1_S222', MAP2K1(S222='p'))
Observable('pMAP2K1_S218', MAP2K1(S218='p'))
Observable('pMAP2K1_S218_S222', MAP2K1(S222='p', S218='p'))
Observable('tMAP2K2', MAP2K2())
Observable('pMAP2K2_S222', MAP2K2(S222='p'))
Observable('pMAP2K2_S226', MAP2K2(S226='p'))
Observable('pMAP2K2_S222_S226', MAP2K2(S222='p', S226='p'))
Observable('tMAPK1', MAPK1())
Observable('pMAPK1_Y187', MAPK1(Y187='p'))
Observable('pMAPK1_T185', MAPK1(T185='p'))
Observable('pMAPK1_T185_Y187', MAPK1(Y187='p', T185='p'))
Observable('tMAPK3', MAPK3())
Observable('pMAPK3_T202', MAPK3(T202='p'))
Observable('pMAPK3_Y204', MAPK3(Y204='p'))
Observable('pMAPK3_T202_Y204', MAPK3(T202='p', Y204='p'))

Expression('inh_MAP2K1', target_MAP2K1/iMEK_kd)
Expression('inh_MAP2K2', target_MAP2K2/iMEK_kd)
Expression('inh_EGFR', target_EGFR/iEGFR_kd)
Expression('EGFR_phosphorylation_Y1173_Y1173_EGF_rate', EGF_obs*EGFR_phosphorylation_Y1173_Y1173_EGF_kcat*INPUT_EGFR_phosphorylation_Y1173_Y1173_EGF_kcat)
Expression('ERBB2_phosphorylation_Y1248_Y1248_EGFR__Y1173_p_rate', EGFR__Y1173_p_obs*ERBB2_phosphorylation_Y1248_Y1248_EGFR__Y1173_p_kcat*INPUT_ERBB2_phosphorylation_Y1248_Y1248_EGFR__Y1173_p_kcat/(inh_EGFR*iEGFR_0 + 1))
Expression('RAF1_phosphorylation_S338_S338_EGFR__Y1173_p_rate', EGFR__Y1173_p_obs*INPUT_RAF1_phosphorylation_S338_S338_EGFR__Y1173_p_kcat*RAF1_phosphorylation_S338_S338_EGFR__Y1173_p_kcat/(inh_EGFR*iEGFR_0 + 1))
Expression('RAF1_phosphorylation_S338_S338_ERBB2__Y1248_p_rate', ERBB2__Y1248_p_obs*INPUT_RAF1_phosphorylation_S338_S338_ERBB2__Y1248_p_kcat*RAF1_phosphorylation_S338_S338_ERBB2__Y1248_p_kcat)
Expression('BRAF_phosphorylation_S445_S445_EGFR__Y1173_p_rate', EGFR__Y1173_p_obs*BRAF_phosphorylation_S445_S445_EGFR__Y1173_p_kcat*INPUT_BRAF_phosphorylation_S445_S445_EGFR__Y1173_p_kcat/(inh_EGFR*iEGFR_0 + 1))
Expression('BRAF_phosphorylation_S445_S445_ERBB2__Y1248_p_rate', ERBB2__Y1248_p_obs*BRAF_phosphorylation_S445_S445_ERBB2__Y1248_p_kcat*INPUT_BRAF_phosphorylation_S445_S445_ERBB2__Y1248_p_kcat)
Expression('MAP2K1_phosphorylation_S218_S222_S218_S222_RAF1__S338_p_rate', RAF1__S338_p_obs*INPUT_MAP2K1_phosphorylation_S218_S222_S218_S222_RAF1__S338_p_kcat*MAP2K1_phosphorylation_S218_S222_S218_S222_RAF1__S338_p_kcat)
Expression('MAP2K1_phosphorylation_S218_S222_S218_S222_BRAF__S445_p_rate', BRAF__S445_p_obs*INPUT_MAP2K1_phosphorylation_S218_S222_S218_S222_BRAF__S445_p_kcat*MAP2K1_phosphorylation_S218_S222_S218_S222_BRAF__S445_p_kcat)
Expression('MAP2K2_phosphorylation_S222_S226_S222_S226_RAF1__S338_p_rate', RAF1__S338_p_obs*INPUT_MAP2K2_phosphorylation_S222_S226_S222_S226_RAF1__S338_p_kcat*MAP2K2_phosphorylation_S222_S226_S222_S226_RAF1__S338_p_kcat)
Expression('MAP2K2_phosphorylation_S222_S226_S222_S226_BRAF__S445_p_rate', BRAF__S445_p_obs*INPUT_MAP2K2_phosphorylation_S222_S226_S222_S226_BRAF__S445_p_kcat*MAP2K2_phosphorylation_S222_S226_S222_S226_BRAF__S445_p_kcat)
Expression('MAPK1_phosphorylation_T185_Y187_T185_Y187_MAP2K1__S218_p__S222_p_rate', MAP2K1__S218_p__S222_p_obs*INPUT_MAPK1_phosphorylation_T185_Y187_T185_Y187_MAP2K1__S218_p__S222_p_kcat*MAPK1_phosphorylation_T185_Y187_T185_Y187_MAP2K1__S218_p__S222_p_kcat/(inh_MAP2K1*iMEK_0 + 1))
Expression('MAPK1_phosphorylation_T185_Y187_T185_Y187_MAP2K2__S222_p__S226_p_rate', MAP2K2__S222_p__S226_p_obs*INPUT_MAPK1_phosphorylation_T185_Y187_T185_Y187_MAP2K2__S222_p__S226_p_kcat*MAPK1_phosphorylation_T185_Y187_T185_Y187_MAP2K2__S222_p__S226_p_kcat/(inh_MAP2K2*iMEK_0 + 1))
Expression('MAPK3_phosphorylation_T202_Y204_T202_Y204_MAP2K1__S218_p__S222_p_rate', MAP2K1__S218_p__S222_p_obs*INPUT_MAPK3_phosphorylation_T202_Y204_T202_Y204_MAP2K1__S218_p__S222_p_kcat*MAPK3_phosphorylation_T202_Y204_T202_Y204_MAP2K1__S218_p__S222_p_kcat/(inh_MAP2K1*iMEK_0 + 1))
Expression('MAPK3_phosphorylation_T202_Y204_T202_Y204_MAP2K2__S222_p__S226_p_rate', MAP2K2__S222_p__S226_p_obs*INPUT_MAPK3_phosphorylation_T202_Y204_T202_Y204_MAP2K2__S222_p__S226_p_kcat*MAPK3_phosphorylation_T202_Y204_T202_Y204_MAP2K2__S222_p__S226_p_kcat/(inh_MAP2K2*iMEK_0 + 1))

Rule('EGF_ext_to_EGF', EGF_ext() | EGF(inh=None), EGF_EGF_kon, EGF_EGF_koff)
Rule('EGFR_Y1173_base', EGFR(Y1173='p') >> EGFR(Y1173='u'), EGFR_dephosphorylation_Y1173_base_rate)
Rule('ERBB2_Y1248_base', ERBB2(Y1248='p') >> ERBB2(Y1248='u'), ERBB2_dephosphorylation_Y1248_base_rate)
Rule('EGFR_phosphorylation_Y1173_Y1173_EGF', EGFR(Y1173='u') >> EGFR(Y1173='p'), EGFR_phosphorylation_Y1173_Y1173_EGF_rate)
Rule('ERBB2_phosphorylation_Y1248_Y1248_EGFR__Y1173_p', ERBB2(Y1248='u') >> ERBB2(Y1248='p'), ERBB2_phosphorylation_Y1248_Y1248_EGFR__Y1173_p_rate)
Rule('RAF1_S338_base', RAF1(S338='p') >> RAF1(S338='u'), RAF1_dephosphorylation_S338_base_rate)
Rule('BRAF_S445_base', BRAF(S445='p') >> BRAF(S445='u'), BRAF_dephosphorylation_S445_base_rate)
Rule('MAP2K1_S222_base', MAP2K1(S222='p') >> MAP2K1(S222='u'), MAP2K1_dephosphorylation_S222_base_rate)
Rule('MAP2K1_S218_base', MAP2K1(S218='p') >> MAP2K1(S218='u'), MAP2K1_dephosphorylation_S218_base_rate)
Rule('MAP2K2_S222_base', MAP2K2(S222='p') >> MAP2K2(S222='u'), MAP2K2_dephosphorylation_S222_base_rate)
Rule('MAP2K2_S226_base', MAP2K2(S226='p') >> MAP2K2(S226='u'), MAP2K2_dephosphorylation_S226_base_rate)
Rule('MAPK1_Y187_base', MAPK1(Y187='p') >> MAPK1(Y187='u'), MAPK1_dephosphorylation_Y187_base_rate)
Rule('MAPK1_T185_base', MAPK1(T185='p') >> MAPK1(T185='u'), MAPK1_dephosphorylation_T185_base_rate)
Rule('MAPK3_T202_base', MAPK3(T202='p') >> MAPK3(T202='u'), MAPK3_dephosphorylation_T202_base_rate)
Rule('MAPK3_Y204_base', MAPK3(Y204='p') >> MAPK3(Y204='u'), MAPK3_dephosphorylation_Y204_base_rate)
Rule('RAF1_phosphorylation_S338_S338_EGFR__Y1173_p', RAF1(S338='u') >> RAF1(S338='p'), RAF1_phosphorylation_S338_S338_EGFR__Y1173_p_rate)
Rule('RAF1_phosphorylation_S338_S338_ERBB2__Y1248_p', RAF1(S338='u') >> RAF1(S338='p'), RAF1_phosphorylation_S338_S338_ERBB2__Y1248_p_rate)
Rule('BRAF_phosphorylation_S445_S445_EGFR__Y1173_p', BRAF(S445='u') >> BRAF(S445='p'), BRAF_phosphorylation_S445_S445_EGFR__Y1173_p_rate)
Rule('BRAF_phosphorylation_S445_S445_ERBB2__Y1248_p', BRAF(S445='u') >> BRAF(S445='p'), BRAF_phosphorylation_S445_S445_ERBB2__Y1248_p_rate)
Rule('MAP2K1_phosphorylation_S218_S222_S218_S222_RAF1__S338_p', MAP2K1(S222='u', S218='u') >> MAP2K1(S222='p', S218='p'), MAP2K1_phosphorylation_S218_S222_S218_S222_RAF1__S338_p_rate)
Rule('MAP2K1_phosphorylation_S218_S222_S218_S222_BRAF__S445_p', MAP2K1(S222='u', S218='u') >> MAP2K1(S222='p', S218='p'), MAP2K1_phosphorylation_S218_S222_S218_S222_BRAF__S445_p_rate)
Rule('MAP2K2_phosphorylation_S222_S226_S222_S226_RAF1__S338_p', MAP2K2(S222='u', S226='u') >> MAP2K2(S222='p', S226='p'), MAP2K2_phosphorylation_S222_S226_S222_S226_RAF1__S338_p_rate)
Rule('MAP2K2_phosphorylation_S222_S226_S222_S226_BRAF__S445_p', MAP2K2(S222='u', S226='u') >> MAP2K2(S222='p', S226='p'), MAP2K2_phosphorylation_S222_S226_S222_S226_BRAF__S445_p_rate)
Rule('MAPK1_phosphorylation_T185_Y187_T185_Y187_MAP2K1__S218_p__S222_p', MAPK1(Y187='u', T185='u') >> MAPK1(Y187='p', T185='p'), MAPK1_phosphorylation_T185_Y187_T185_Y187_MAP2K1__S218_p__S222_p_rate)
Rule('MAPK1_phosphorylation_T185_Y187_T185_Y187_MAP2K2__S222_p__S226_p', MAPK1(Y187='u', T185='u') >> MAPK1(Y187='p', T185='p'), MAPK1_phosphorylation_T185_Y187_T185_Y187_MAP2K2__S222_p__S226_p_rate)
Rule('MAPK3_phosphorylation_T202_Y204_T202_Y204_MAP2K1__S218_p__S222_p', MAPK3(T202='u', Y204='u') >> MAPK3(T202='p', Y204='p'), MAPK3_phosphorylation_T202_Y204_T202_Y204_MAP2K1__S218_p__S222_p_rate)
Rule('MAPK3_phosphorylation_T202_Y204_T202_Y204_MAP2K2__S222_p__S226_p', MAPK3(T202='u', Y204='u') >> MAPK3(T202='p', Y204='p'), MAPK3_phosphorylation_T202_Y204_T202_Y204_MAP2K2__S222_p__S226_p_rate)

Initial(EGF(inh=None), EGF_init)
Initial(EGF_ext(), EGF_0, fixed=True)
Initial(EGFR(Y1173='u', inh=None), EGFR_init)
Initial(ERBB2(Y1248='u', inh=None), ERBB2_init)
Initial(RAF1(S338='u', inh=None), RAF1_init)
Initial(BRAF(S445='u', inh=None), BRAF_init)
Initial(MAP2K1(S222='u', S218='u', inh=None), MAP2K1_init)
Initial(MAP2K2(S222='u', S226='u', inh=None), MAP2K2_init)
Initial(MAPK1(Y187='u', T185='u', inh=None), MAPK1_init)
Initial(MAPK3(T202='u', Y204='u', inh=None), MAPK3_init)

