# exported from PySB model 'EGFR_MAPK'

from pysb import Model, Monomer, Parameter, Expression, Compartment, Rule, Observable, Initial, MatchOnce, Annotation, MultiState, Tag, ANY, WILD

Model()

Monomer('EGF', ['inh'])
Monomer('EGF_ext')
Monomer('EGFR', ['Y1173', 'inh'], {'Y1173': ['u', 'p']})
Monomer('ERBB2', ['Y1248', 'inh'], {'Y1248': ['u', 'p']})
Monomer('MAP2K1', ['S218', 'S222', 'inh'], {'S218': ['u', 'p'], 'S222': ['u', 'p']})
Monomer('MAP2K2', ['S222', 'S226', 'inh'], {'S222': ['u', 'p'], 'S226': ['u', 'p']})
Monomer('MAPK1', ['T185', 'Y187', 'inh'], {'T185': ['u', 'p'], 'Y187': ['u', 'p']})
Monomer('MAPK3', ['T202', 'Y204', 'inh'], {'T202': ['u', 'p'], 'Y204': ['u', 'p']})
Monomer('RPS6KA1', ['S380', 'inh'], {'S380': ['u', 'p']})

Parameter('EGF_eq', 100.0)
Parameter('INPUT_EGF_eq', 0.0)
Parameter('EGF_0', 0.0)
Parameter('EGF_EGF_koff', 0.1)
Parameter('EGF_EGF_kd', 1.0)
Parameter('EGFR_eq', 100.0)
Parameter('INPUT_EGFR_eq', 0.0)
Parameter('EGFR_degradation_kdeg', 1.0)
Parameter('INPUT_EGFR_degradation_kdeg', 0.0)
Parameter('EGFR_dephosphorylation_Y1173_base_kcat', 1.0)
Parameter('INPUT_EGFR_dephosphorylation_Y1173_base_kcat', 0.0)
Parameter('ERBB2_eq', 100.0)
Parameter('INPUT_ERBB2_eq', 0.0)
Parameter('ERBB2_degradation_kdeg', 1.0)
Parameter('INPUT_ERBB2_degradation_kdeg', 0.0)
Parameter('ERBB2_dephosphorylation_Y1248_base_kcat', 1.0)
Parameter('INPUT_ERBB2_dephosphorylation_Y1248_base_kcat', 0.0)
Parameter('EGFR_phosphorylation_Y1173_base_kcat', 1.0)
Parameter('INPUT_EGFR_phosphorylation_Y1173_base_kcat', 0.0)
Parameter('EGFR_activation_Y1173_EGF_kw', 1.0)
Parameter('INPUT_EGFR_activation_Y1173_EGF_kw', 0.0)
Parameter('ERBB2_phosphorylation_Y1248_base_kcat', 1.0)
Parameter('INPUT_ERBB2_phosphorylation_Y1248_base_kcat', 0.0)
Parameter('ERBB2_activation_Y1248_EGFR__Y1173_p_kw', 1.0)
Parameter('INPUT_ERBB2_activation_Y1248_EGFR__Y1173_p_kw', 0.0)
Parameter('degradation_EGFR__Y1173_p_kdeg', 0.0)
Parameter('INPUT_degradation_EGFR__Y1173_p_kdeg', 0.0)
Parameter('degradation_ERBB2__Y1248_p_kdeg', 0.0)
Parameter('INPUT_degradation_ERBB2__Y1248_p_kdeg', 0.0)
Parameter('MAP2K1_eq', 100.0)
Parameter('INPUT_MAP2K1_eq', 0.0)
Parameter('MAP2K1_dephosphorylation_S218_base_kcat', 1.0)
Parameter('INPUT_MAP2K1_dephosphorylation_S218_base_kcat', 0.0)
Parameter('MAP2K1_dephosphorylation_S222_base_kcat', 1.0)
Parameter('INPUT_MAP2K1_dephosphorylation_S222_base_kcat', 0.0)
Parameter('MAP2K2_eq', 100.0)
Parameter('INPUT_MAP2K2_eq', 0.0)
Parameter('MAP2K2_dephosphorylation_S226_base_kcat', 1.0)
Parameter('INPUT_MAP2K2_dephosphorylation_S226_base_kcat', 0.0)
Parameter('MAP2K2_dephosphorylation_S222_base_kcat', 1.0)
Parameter('INPUT_MAP2K2_dephosphorylation_S222_base_kcat', 0.0)
Parameter('MAPK1_eq', 100.0)
Parameter('INPUT_MAPK1_eq', 0.0)
Parameter('MAPK1_dephosphorylation_T185_base_kcat', 1.0)
Parameter('INPUT_MAPK1_dephosphorylation_T185_base_kcat', 0.0)
Parameter('MAPK1_dephosphorylation_Y187_base_kcat', 1.0)
Parameter('INPUT_MAPK1_dephosphorylation_Y187_base_kcat', 0.0)
Parameter('MAPK3_eq', 100.0)
Parameter('INPUT_MAPK3_eq', 0.0)
Parameter('MAPK3_dephosphorylation_T202_base_kcat', 1.0)
Parameter('INPUT_MAPK3_dephosphorylation_T202_base_kcat', 0.0)
Parameter('MAPK3_dephosphorylation_Y204_base_kcat', 1.0)
Parameter('INPUT_MAPK3_dephosphorylation_Y204_base_kcat', 0.0)
Parameter('RPS6KA1_eq', 100.0)
Parameter('INPUT_RPS6KA1_eq', 0.0)
Parameter('RPS6KA1_dephosphorylation_S380_base_kcat', 1.0)
Parameter('INPUT_RPS6KA1_dephosphorylation_S380_base_kcat', 0.0)
Parameter('MAP2K1_phosphorylation_S218_S222_base_kcat', 1.0)
Parameter('INPUT_MAP2K1_phosphorylation_S218_S222_base_kcat', 0.0)
Parameter('MAP2K1_activation_S218_S222_EGFR__Y1173_p_kw', 1.0)
Parameter('INPUT_MAP2K1_activation_S218_S222_EGFR__Y1173_p_kw', 0.0)
Parameter('MAP2K1_activation_S218_S222_ERBB2__Y1248_p_kw', 1.0)
Parameter('INPUT_MAP2K1_activation_S218_S222_ERBB2__Y1248_p_kw', 0.0)
Parameter('MAP2K1_deactivation_S218_S222_MAPK1__T185_p__Y187_p_kw', 1.0)
Parameter('INPUT_MAP2K1_deactivation_S218_S222_MAPK1__T185_p__Y187_p_kw', 0.0)
Parameter('MAP2K1_deactivation_S218_S222_MAPK3__T202_p__Y204_p_kw', 1.0)
Parameter('INPUT_MAP2K1_deactivation_S218_S222_MAPK3__T202_p__Y204_p_kw', 0.0)
Parameter('MAP2K2_phosphorylation_S222_S226_base_kcat', 1.0)
Parameter('INPUT_MAP2K2_phosphorylation_S222_S226_base_kcat', 0.0)
Parameter('MAP2K2_activation_S222_S226_EGFR__Y1173_p_kw', 1.0)
Parameter('INPUT_MAP2K2_activation_S222_S226_EGFR__Y1173_p_kw', 0.0)
Parameter('MAP2K2_activation_S222_S226_ERBB2__Y1248_p_kw', 1.0)
Parameter('INPUT_MAP2K2_activation_S222_S226_ERBB2__Y1248_p_kw', 0.0)
Parameter('MAP2K2_deactivation_S222_S226_MAPK1__T185_p__Y187_p_kw', 1.0)
Parameter('INPUT_MAP2K2_deactivation_S222_S226_MAPK1__T185_p__Y187_p_kw', 0.0)
Parameter('MAP2K2_deactivation_S222_S226_MAPK3__T202_p__Y204_p_kw', 1.0)
Parameter('INPUT_MAP2K2_deactivation_S222_S226_MAPK3__T202_p__Y204_p_kw', 0.0)
Parameter('MAPK1_phosphorylation_T185_Y187_base_kcat', 1.0)
Parameter('INPUT_MAPK1_phosphorylation_T185_Y187_base_kcat', 0.0)
Parameter('MAPK1_activation_T185_Y187_MAP2K1__S218_p__S222_p_kw', 1.0)
Parameter('INPUT_MAPK1_activation_T185_Y187_MAP2K1__S218_p__S222_p_kw', 0.0)
Parameter('MAPK1_activation_T185_Y187_MAP2K2__S222_p__S226_p_kw', 1.0)
Parameter('INPUT_MAPK1_activation_T185_Y187_MAP2K2__S222_p__S226_p_kw', 0.0)
Parameter('MAPK3_phosphorylation_T202_Y204_base_kcat', 1.0)
Parameter('INPUT_MAPK3_phosphorylation_T202_Y204_base_kcat', 0.0)
Parameter('MAPK3_activation_T202_Y204_MAP2K1__S218_p__S222_p_kw', 1.0)
Parameter('INPUT_MAPK3_activation_T202_Y204_MAP2K1__S218_p__S222_p_kw', 0.0)
Parameter('MAPK3_activation_T202_Y204_MAP2K2__S222_p__S226_p_kw', 1.0)
Parameter('INPUT_MAPK3_activation_T202_Y204_MAP2K2__S222_p__S226_p_kw', 0.0)
Parameter('RPS6KA1_phosphorylation_S380_base_kcat', 1.0)
Parameter('INPUT_RPS6KA1_phosphorylation_S380_base_kcat', 0.0)
Parameter('RPS6KA1_activation_S380_MAPK1__T185_p__Y187_p_kw', 1.0)
Parameter('INPUT_RPS6KA1_activation_S380_MAPK1__T185_p__Y187_p_kw', 0.0)
Parameter('RPS6KA1_activation_S380_MAPK3__T202_p__Y204_p_kw', 1.0)
Parameter('INPUT_RPS6KA1_activation_S380_MAPK3__T202_p__Y204_p_kw', 0.0)
Parameter('iMEK_0', 0.0)
Parameter('iMEK_MAP2K1_kd', 0.0)
Parameter('INPUT_iMEK_MAP2K1_kd', 0.0)
Parameter('iMEK_MAP2K2_kd', 0.0)
Parameter('INPUT_iMEK_MAP2K2_kd', 0.0)
Parameter('iEGFR_0', 0.0)
Parameter('iEGFR_EGFR_kd', 0.0)
Parameter('INPUT_iEGFR_EGFR_kd', 0.0)
Parameter('iPI3K_0', 0.0)
Parameter('iPKC_0', 0.0)

Expression('EGF_init', EGF_eq*INPUT_EGF_eq)
Expression('EGF_EGF_kon', EGF_EGF_kd*EGF_EGF_koff)
Expression('EGFR_init', EGFR_eq*INPUT_EGFR_eq)
Expression('EGFR_synthesis_ksyn', EGFR_init*EGFR_degradation_kdeg*INPUT_EGFR_degradation_kdeg)
Expression('EGFR_dephosphorylation_Y1173_base_rate', EGFR_dephosphorylation_Y1173_base_kcat*INPUT_EGFR_dephosphorylation_Y1173_base_kcat)
Expression('ERBB2_init', ERBB2_eq*INPUT_ERBB2_eq)
Expression('ERBB2_synthesis_ksyn', ERBB2_init*ERBB2_degradation_kdeg*INPUT_ERBB2_degradation_kdeg)
Expression('ERBB2_dephosphorylation_Y1248_base_rate', ERBB2_dephosphorylation_Y1248_base_kcat*INPUT_ERBB2_dephosphorylation_Y1248_base_kcat)
Expression('degradation_EGFR__Y1173_p_rate', INPUT_degradation_EGFR__Y1173_p_kdeg*degradation_EGFR__Y1173_p_kdeg)
Expression('degradation_ERBB2__Y1248_p_rate', INPUT_degradation_ERBB2__Y1248_p_kdeg*degradation_ERBB2__Y1248_p_kdeg)
Expression('MAP2K1_init', INPUT_MAP2K1_eq*MAP2K1_eq)
Expression('MAP2K1_dephosphorylation_S218_base_rate', INPUT_MAP2K1_dephosphorylation_S218_base_kcat*MAP2K1_dephosphorylation_S218_base_kcat)
Expression('MAP2K1_dephosphorylation_S222_base_rate', INPUT_MAP2K1_dephosphorylation_S222_base_kcat*MAP2K1_dephosphorylation_S222_base_kcat)
Expression('MAP2K2_init', INPUT_MAP2K2_eq*MAP2K2_eq)
Expression('MAP2K2_dephosphorylation_S226_base_rate', INPUT_MAP2K2_dephosphorylation_S226_base_kcat*MAP2K2_dephosphorylation_S226_base_kcat)
Expression('MAP2K2_dephosphorylation_S222_base_rate', INPUT_MAP2K2_dephosphorylation_S222_base_kcat*MAP2K2_dephosphorylation_S222_base_kcat)
Expression('MAPK1_init', INPUT_MAPK1_eq*MAPK1_eq)
Expression('MAPK1_dephosphorylation_T185_base_rate', INPUT_MAPK1_dephosphorylation_T185_base_kcat*MAPK1_dephosphorylation_T185_base_kcat)
Expression('MAPK1_dephosphorylation_Y187_base_rate', INPUT_MAPK1_dephosphorylation_Y187_base_kcat*MAPK1_dephosphorylation_Y187_base_kcat)
Expression('MAPK3_init', INPUT_MAPK3_eq*MAPK3_eq)
Expression('MAPK3_dephosphorylation_T202_base_rate', INPUT_MAPK3_dephosphorylation_T202_base_kcat*MAPK3_dephosphorylation_T202_base_kcat)
Expression('MAPK3_dephosphorylation_Y204_base_rate', INPUT_MAPK3_dephosphorylation_Y204_base_kcat*MAPK3_dephosphorylation_Y204_base_kcat)
Expression('RPS6KA1_init', INPUT_RPS6KA1_eq*RPS6KA1_eq)
Expression('RPS6KA1_dephosphorylation_S380_base_rate', INPUT_RPS6KA1_dephosphorylation_S380_base_kcat*RPS6KA1_dephosphorylation_S380_base_kcat)

Observable('EGF_obs', EGF(inh=None))
Observable('EGFR__Y1173_p_obs', EGFR(Y1173='p', inh=None))
Observable('ERBB2__Y1248_p_obs', ERBB2(Y1248='p', inh=None))
Observable('MAPK1__T185_p__Y187_p_obs', MAPK1(T185='p', Y187='p', inh=None))
Observable('MAPK3__T202_p__Y204_p_obs', MAPK3(T202='p', Y204='p', inh=None))
Observable('MAP2K1__S218_p__S222_p_obs', MAP2K1(S218='p', S222='p', inh=None))
Observable('MAP2K2__S222_p__S226_p_obs', MAP2K2(S222='p', S226='p', inh=None))
Observable('ERK_T202_Y204', MAPK1(T185='p', Y187='p') + MAPK3(T202='p', Y204='p'))
Observable('MEK_S221', MAP2K1(S222='p') + MAP2K2(S226='p'))
Observable('tEGF', EGF())
Observable('tEGF_ext', EGF_ext())
Observable('tEGFR', EGFR())
Observable('pEGFR_Y1173', EGFR(Y1173='p'))
Observable('tERBB2', ERBB2())
Observable('pERBB2_Y1248', ERBB2(Y1248='p'))
Observable('tMAP2K1', MAP2K1())
Observable('pMAP2K1_S218', MAP2K1(S218='p'))
Observable('pMAP2K1_S222', MAP2K1(S222='p'))
Observable('pMAP2K1_S218_S222', MAP2K1(S218='p', S222='p'))
Observable('tMAP2K2', MAP2K2())
Observable('pMAP2K2_S222', MAP2K2(S222='p'))
Observable('pMAP2K2_S226', MAP2K2(S226='p'))
Observable('pMAP2K2_S222_S226', MAP2K2(S222='p', S226='p'))
Observable('tMAPK1', MAPK1())
Observable('pMAPK1_T185', MAPK1(T185='p'))
Observable('pMAPK1_Y187', MAPK1(Y187='p'))
Observable('pMAPK1_T185_Y187', MAPK1(T185='p', Y187='p'))
Observable('tMAPK3', MAPK3())
Observable('pMAPK3_T202', MAPK3(T202='p'))
Observable('pMAPK3_Y204', MAPK3(Y204='p'))
Observable('pMAPK3_T202_Y204', MAPK3(T202='p', Y204='p'))
Observable('tRPS6KA1', RPS6KA1())
Observable('pRPS6KA1_S380', RPS6KA1(S380='p'))
Observable('target_MAP2K1', MAP2K1())
Observable('target_MAP2K2', MAP2K2())
Observable('target_EGFR', EGFR())

Expression('inh_EGFR', target_EGFR/(INPUT_iEGFR_EGFR_kd*iEGFR_EGFR_kd))
Expression('inh_MAP2K1', target_MAP2K1/(INPUT_iMEK_MAP2K1_kd*iMEK_MAP2K1_kd))
Expression('inh_MAP2K2', target_MAP2K2/(INPUT_iMEK_MAP2K2_kd*iMEK_MAP2K2_kd))
Expression('EGFR_phosphorylation_Y1173_activation_rate', 1.0*EGF_obs*EGFR_activation_Y1173_EGF_kw*EGFR_phosphorylation_Y1173_base_kcat*INPUT_EGFR_activation_Y1173_EGF_kw)
Expression('ERBB2_phosphorylation_Y1248_activation_rate', 1.0*EGFR__Y1173_p_obs*ERBB2_activation_Y1248_EGFR__Y1173_p_kw*ERBB2_phosphorylation_Y1248_base_kcat*INPUT_ERBB2_activation_Y1248_EGFR__Y1173_p_kw/(inh_EGFR*iEGFR_0 + 1))
Expression('MAP2K1_phosphorylation_S218_S222_activation_rate', MAP2K1_phosphorylation_S218_S222_base_kcat*(EGFR__Y1173_p_obs*INPUT_MAP2K1_activation_S218_S222_EGFR__Y1173_p_kw*MAP2K1_activation_S218_S222_EGFR__Y1173_p_kw/(inh_EGFR*iEGFR_0 + 1) + ERBB2__Y1248_p_obs*INPUT_MAP2K1_activation_S218_S222_ERBB2__Y1248_p_kw*MAP2K1_activation_S218_S222_ERBB2__Y1248_p_kw)/(MAPK1__T185_p__Y187_p_obs*INPUT_MAP2K1_deactivation_S218_S222_MAPK1__T185_p__Y187_p_kw*MAP2K1_deactivation_S218_S222_MAPK1__T185_p__Y187_p_kw + MAPK3__T202_p__Y204_p_obs*INPUT_MAP2K1_deactivation_S218_S222_MAPK3__T202_p__Y204_p_kw*MAP2K1_deactivation_S218_S222_MAPK3__T202_p__Y204_p_kw + 1.0))
Expression('MAP2K2_phosphorylation_S222_S226_activation_rate', MAP2K2_phosphorylation_S222_S226_base_kcat*(EGFR__Y1173_p_obs*INPUT_MAP2K2_activation_S222_S226_EGFR__Y1173_p_kw*MAP2K2_activation_S222_S226_EGFR__Y1173_p_kw/(inh_EGFR*iEGFR_0 + 1) + ERBB2__Y1248_p_obs*INPUT_MAP2K2_activation_S222_S226_ERBB2__Y1248_p_kw*MAP2K2_activation_S222_S226_ERBB2__Y1248_p_kw)/(MAPK1__T185_p__Y187_p_obs*INPUT_MAP2K2_deactivation_S222_S226_MAPK1__T185_p__Y187_p_kw*MAP2K2_deactivation_S222_S226_MAPK1__T185_p__Y187_p_kw + MAPK3__T202_p__Y204_p_obs*INPUT_MAP2K2_deactivation_S222_S226_MAPK3__T202_p__Y204_p_kw*MAP2K2_deactivation_S222_S226_MAPK3__T202_p__Y204_p_kw + 1.0))
Expression('MAPK1_phosphorylation_T185_Y187_activation_rate', 1.0*MAPK1_phosphorylation_T185_Y187_base_kcat*(MAP2K1__S218_p__S222_p_obs*INPUT_MAPK1_activation_T185_Y187_MAP2K1__S218_p__S222_p_kw*MAPK1_activation_T185_Y187_MAP2K1__S218_p__S222_p_kw + MAP2K2__S222_p__S226_p_obs*INPUT_MAPK1_activation_T185_Y187_MAP2K2__S222_p__S226_p_kw*MAPK1_activation_T185_Y187_MAP2K2__S222_p__S226_p_kw/(inh_MAP2K2*iMEK_0 + 1)))
Expression('MAPK3_phosphorylation_T202_Y204_activation_rate', 1.0*MAPK3_phosphorylation_T202_Y204_base_kcat*(MAP2K1__S218_p__S222_p_obs*INPUT_MAPK3_activation_T202_Y204_MAP2K1__S218_p__S222_p_kw*MAPK3_activation_T202_Y204_MAP2K1__S218_p__S222_p_kw/(inh_MAP2K1*iMEK_0 + 1) + MAP2K2__S222_p__S226_p_obs*INPUT_MAPK3_activation_T202_Y204_MAP2K2__S222_p__S226_p_kw*MAPK3_activation_T202_Y204_MAP2K2__S222_p__S226_p_kw))
Expression('RPS6KA1_phosphorylation_S380_activation_rate', 1.0*RPS6KA1_phosphorylation_S380_base_kcat*(MAPK1__T185_p__Y187_p_obs*INPUT_RPS6KA1_activation_S380_MAPK1__T185_p__Y187_p_kw*RPS6KA1_activation_S380_MAPK1__T185_p__Y187_p_kw + MAPK3__T202_p__Y204_p_obs*INPUT_RPS6KA1_activation_S380_MAPK3__T202_p__Y204_p_kw*RPS6KA1_activation_S380_MAPK3__T202_p__Y204_p_kw))

Rule('EGF_ext_to_EGF', EGF_ext() | EGF(inh=None), EGF_EGF_kon, EGF_EGF_koff)
Rule('synthesis_EGFR', None >> EGFR(Y1173='u', inh=None), EGFR_synthesis_ksyn)
Rule('degradation_EGFR', EGFR() >> None, EGFR_degradation_kdeg)
Rule('EGFR_Y1173_base', EGFR(Y1173='p') >> EGFR(Y1173='u'), EGFR_dephosphorylation_Y1173_base_rate)
Rule('synthesis_ERBB2', None >> ERBB2(Y1248='u', inh=None), ERBB2_synthesis_ksyn)
Rule('degradation_ERBB2', ERBB2() >> None, ERBB2_degradation_kdeg)
Rule('ERBB2_Y1248_base', ERBB2(Y1248='p') >> ERBB2(Y1248='u'), ERBB2_dephosphorylation_Y1248_base_rate)
Rule('EGFR_phosphorylation_Y1173_activation', EGFR(Y1173='u') >> EGFR(Y1173='p'), EGFR_phosphorylation_Y1173_activation_rate)
Rule('ERBB2_phosphorylation_Y1248_activation', ERBB2(Y1248='u') >> ERBB2(Y1248='p'), ERBB2_phosphorylation_Y1248_activation_rate)
Rule('degradation_EGFR__Y1173_p', EGFR(Y1173='p') >> None, degradation_EGFR__Y1173_p_rate)
Rule('degradation_ERBB2__Y1248_p', ERBB2(Y1248='p') >> None, degradation_ERBB2__Y1248_p_rate)
Rule('MAP2K1_S218_base', MAP2K1(S218='p') >> MAP2K1(S218='u'), MAP2K1_dephosphorylation_S218_base_rate)
Rule('MAP2K1_S222_base', MAP2K1(S222='p') >> MAP2K1(S222='u'), MAP2K1_dephosphorylation_S222_base_rate)
Rule('MAP2K2_S226_base', MAP2K2(S226='p') >> MAP2K2(S226='u'), MAP2K2_dephosphorylation_S226_base_rate)
Rule('MAP2K2_S222_base', MAP2K2(S222='p') >> MAP2K2(S222='u'), MAP2K2_dephosphorylation_S222_base_rate)
Rule('MAPK1_T185_base', MAPK1(T185='p') >> MAPK1(T185='u'), MAPK1_dephosphorylation_T185_base_rate)
Rule('MAPK1_Y187_base', MAPK1(Y187='p') >> MAPK1(Y187='u'), MAPK1_dephosphorylation_Y187_base_rate)
Rule('MAPK3_T202_base', MAPK3(T202='p') >> MAPK3(T202='u'), MAPK3_dephosphorylation_T202_base_rate)
Rule('MAPK3_Y204_base', MAPK3(Y204='p') >> MAPK3(Y204='u'), MAPK3_dephosphorylation_Y204_base_rate)
Rule('RPS6KA1_S380_base', RPS6KA1(S380='p') >> RPS6KA1(S380='u'), RPS6KA1_dephosphorylation_S380_base_rate)
Rule('MAP2K1_phosphorylation_S218_S222_activation', MAP2K1(S218='u', S222='u') >> MAP2K1(S218='p', S222='p'), MAP2K1_phosphorylation_S218_S222_activation_rate)
Rule('MAP2K2_phosphorylation_S222_S226_activation', MAP2K2(S222='u', S226='u') >> MAP2K2(S222='p', S226='p'), MAP2K2_phosphorylation_S222_S226_activation_rate)
Rule('MAPK1_phosphorylation_T185_Y187_activation', MAPK1(T185='u', Y187='u') >> MAPK1(T185='p', Y187='p'), MAPK1_phosphorylation_T185_Y187_activation_rate)
Rule('MAPK3_phosphorylation_T202_Y204_activation', MAPK3(T202='u', Y204='u') >> MAPK3(T202='p', Y204='p'), MAPK3_phosphorylation_T202_Y204_activation_rate)
Rule('RPS6KA1_phosphorylation_S380_activation', RPS6KA1(S380='u') >> RPS6KA1(S380='p'), RPS6KA1_phosphorylation_S380_activation_rate)

Initial(EGF(inh=None), EGF_init)
Initial(EGF_ext(), EGF_0, fixed=True)
Initial(EGFR(Y1173='u', inh=None), EGFR_init)
Initial(ERBB2(Y1248='u', inh=None), ERBB2_init)
Initial(MAP2K1(S218='u', S222='u', inh=None), MAP2K1_init)
Initial(MAP2K2(S222='u', S226='u', inh=None), MAP2K2_init)
Initial(MAPK1(T185='u', Y187='u', inh=None), MAPK1_init)
Initial(MAPK3(T202='u', Y204='u', inh=None), MAPK3_init)
Initial(RPS6KA1(S380='u', inh=None), RPS6KA1_init)

