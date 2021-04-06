# exported from PySB model 'EGFR'

from pysb import Model, Monomer, Parameter, Expression, Compartment, Rule, Observable, Initial, MatchOnce, Annotation, MultiState, Tag, ANY, WILD

Model()

Monomer('EGF', ['inh'])
Monomer('EGF_ext')
Monomer('ERBB2', ['Y1248', 'inh'], {'Y1248': ['u', 'p']})

Parameter('EGF_degradation_kdeg', 1.0)
Parameter('EGF_eq', 100.0)
Parameter('INPUT_EGF_eq', 0.0)
Parameter('EGF_0', 0.0)
Parameter('EGF_EGF_koff', 0.1)
Parameter('EGF_EGF_kd', 1.0)
Parameter('ERBB2_degradation_kdeg', 1.0)
Parameter('ERBB2_eq', 100.0)
Parameter('INPUT_ERBB2_eq', 0.0)
Parameter('ERBB2_phosphorylation_Y1248_base_kcat', 1.0)
Parameter('ERBB2_dephosphorylation_Y1248_base_kcat', 1.0)
Parameter('INPUT_ERBB2_phosphorylation_Y1248_base_kcat', 0.0)
Parameter('INPUT_ERBB2_dephosphorylation_Y1248_base_kcat', 0.0)
Parameter('ERBB2_phosphorylation_Y1248_EGF_kcat', 1.0)
Parameter('INPUT_ERBB2_phosphorylation_Y1248_EGF_kcat', 0.0)

Expression('EGF_synthesis_ksyn', EGF_degradation_kdeg*EGF_eq)
Expression('EGF_synthesis_rate', EGF_synthesis_ksyn*INPUT_EGF_eq)
Expression('EGF_degradation_rate', EGF_degradation_kdeg)
Expression('EGF_ss', EGF_synthesis_rate/EGF_degradation_rate)
Expression('EGF_EGF_kon', EGF_EGF_kd*EGF_EGF_koff)
Expression('ERBB2_synthesis_ksyn', ERBB2_degradation_kdeg*ERBB2_eq)
Expression('ERBB2_synthesis_rate', ERBB2_synthesis_ksyn*INPUT_ERBB2_eq)
Expression('ERBB2_degradation_rate', ERBB2_degradation_kdeg)
Expression('ERBB2_ss', ERBB2_synthesis_rate/ERBB2_degradation_rate)
Expression('ERBB2_phosphorylation_Y1248_base_rate', ERBB2_phosphorylation_Y1248_base_kcat*INPUT_ERBB2_phosphorylation_Y1248_base_kcat)
Expression('ERBB2_dephosphorylation_Y1248_base_rate', ERBB2_dephosphorylation_Y1248_base_kcat*INPUT_ERBB2_dephosphorylation_Y1248_base_kcat)

Observable('EGF_obs', EGF(inh=None))
Observable('tEGF', EGF())
Observable('tEGF_ext', EGF_ext())
Observable('tERBB2', ERBB2())
Observable('pERBB2_Y1248', ERBB2(Y1248='p'))

Expression('ERBB2_phosphorylation_Y1248_EGF_rate', EGF_obs*ERBB2_phosphorylation_Y1248_EGF_kcat*INPUT_ERBB2_phosphorylation_Y1248_EGF_kcat)

Rule('EGF_ext_to_EGF', EGF_ext() | EGF(inh=None), EGF_EGF_kon, EGF_EGF_koff)
Rule('ERBB2_Y1248_base', ERBB2(Y1248='p') >> ERBB2(Y1248='u'), ERBB2_dephosphorylation_Y1248_base_rate)
Rule('ERBB2_phosphorylation_Y1248_EGF', ERBB2(Y1248='u') >> ERBB2(Y1248='p'), ERBB2_phosphorylation_Y1248_EGF_rate)

Initial(EGF(inh=None), EGF_ss)
Initial(EGF_ext(), EGF_0, fixed=True)
Initial(ERBB2(Y1248='u', inh=None), ERBB2_ss)

