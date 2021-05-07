import os

from mEncoder import data_dir, pretrain_dir, results_dir

HIDDEN_LAYERS = [2, 3, 5]
PATHWAYS = ['EGFR', 'EGFR_MAPK', 'EGFR_MAPK_AKT', 'EGFR_MAPK_AKT_STAT',
            'EGFR_MAPK_AKT_STAT_S6']
DATASETS = ['dream_cytof']

SAMPLES = ['c184A1', 'cBT20', 'cBT474', 'cBT549', 'cCAL148', 'cCAL851',
           'cCAL51', 'cDU4475', 'cEFM192A', 'cEVSAT', 'HBL100', 'HCC1187',
           'HCC1395', 'HCC1419', 'HCC1500', 'HCC1569', 'HCC1599', 'HCC1937',
           'HCC1954', 'HCC2157', 'HCC2185', 'HCC3153', 'HCC38', 'HCC70',
           'HDQP1', 'JIMT1', 'MCF10A', 'MCF10F', 'MCF7', 'MDAMB134VI',
           'MDAMB157', 'MDAMB175VII', 'MDAMB361', 'MDAMB415', 'MDAMB453',
           'MDAkb2', 'MFM223', 'MPE600', 'MX1', 'OCUBM', 'T47D', 'UACC812',
           'UACC893', 'ZR7530']
samplestr = '.'.join(SAMPLES)

STARTS = [str(i) for i in range(int(config["num_starts"]))]

rule process_data:
    input:
        script='process_data.py',
        data_code=os.path.join('mEncoder', 'generate_data.py'),
        enc_code=os.path.join('mEncoder', 'encoder.py'),
        model_code=os.path.join('mEncoder', 'mechanistic_model.py'),
        pathway=os.path.join('mEncoder', 'pathways.py')
    output:
        datafiles=expand(
            os.path.join(data_dir, '{{data}}__{{model}}__{file}.tsv'),
            file=['conditions', 'measurements', 'observables']
        )
    wildcard_constraints:
        model='[\w_]+',
        data='[\w]+',
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data}'

rule compile_mechanistic_model:
    input:
        script='compile_model.py',
        model_code=os.path.join('mEncoder', 'mechanistic_model.py'),
        enc_code=os.path.join('mEncoder', 'encoder.py'),
        autoencoder_code=os.path.join('mEncoder', 'autoencoder.py'),
        pathway=os.path.join('pathways', 'pw_{model}.py'),
        pathways=os.path.join('mEncoder', 'pathways.py'),
        data=rules.process_data.output.datafiles
    output:
        model=os.path.join('amici_models', '{model}_{data}__{model}_petab',
                           '{model}', '{model}.py'),
    wildcard_constraints:
        model='[\w_]+',
        data='[\w]+',
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data}'


rule pretrain_per_sample:
    input:
        script='pretrain_per_sample.py',
        pretraining_code=os.path.join('mEncoder', 'pretraining.py'),
        model_code=os.path.join('mEncoder', 'mechanistic_model.py'),
        model=rules.compile_mechanistic_model.output.model,
        data=rules.process_data.output.datafiles,
    output:
        pretraining=os.path.join(
            pretrain_dir, '{model}__{data}__{model}__{sample}.csv'
        ),
        result=os.path.join(
            pretrain_dir, '{model}__{data}__{model}__{sample}.hdf5'
        )
    wildcard_constraints:
        model='[\w_]+',
        data='[\w]+',
        sample='[\w_]+',
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data} '
        '{wildcards.sample}'


rule pretrain_cross_sample:
    input:
        script='pretrain_cross_samples.py',
        pretraining_code=os.path.join('mEncoder', 'pretraining.py'),
        model_code=os.path.join('mEncoder', 'mechanistic_model.py'),
        pretrain_per_sample=expand(
            os.path.join(
                'pretraining', '{{model}}__{{data}}__{{model}}__{sample}.csv'
            ),
            sample=SAMPLES
        ),
        model=rules.compile_mechanistic_model.output.model,
        data=rules.process_data.output.datafiles,
    output:
        pretraining=os.path.join(
            'pretraining',
            f'{{model}}__{{data}}__{{model}}__{samplestr}__pca__'
            f'{{n_hidden}}__{{job}}.hdf5'
        )
    wildcard_constraints:
        model='[\w_]+',
        data='[\w]+',
        n_hidden='[0-9]+',
        job='[0-9]+',
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data} '
        '{samplestr} pca {wildcards.n_hidden} {wildcards.job}'


rule estimate_parameters:
    input:
        script='run_estimation.py',
        encoder_code=os.path.join('mEncoder', 'encoder.py'),
        training_code=os.path.join('mEncoder', 'training.py'),
        autoencoder_code=os.path.join('mEncoder', 'autoencoder.py'),
        dataset=rules.process_data.output.datafiles,
        pretrain_encoder=rules.pretrain_cross_sample.output.pretraining,
        model=rules.compile_mechanistic_model.output.model,
    output:
        result=os.path.join(results_dir, '{model}', '{data}',
                            samplestr + '__{n_hidden}__{job}.hdf5'),
    wildcard_constraints:
        model='[\w_]+',
        data='[\w]+',
        n_hidden='[0-9]+',
        job='[0-9]+',
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data} '
        '{samplestr} {wildcards.n_hidden} {wildcards.job}'

rule collect_estimation_results:
    input:
        script='collect_estimation.py',
        trace=expand(os.path.join(
            results_dir, '{{model}}', '{{data}}',
            samplestr + '__{{n_hidden}}__{job}.pickle'
        ), job=STARTS)
    output:
        result=os.path.join(results_dir, '{model}', '{data}',
                            samplestr + '__{n_hidden}__full.hdf5'),
    wildcard_constraints:
        model='[\w_]+',
        data='[\w_]+',
        n_hidden='[0-9]+',
        job='[0-9]+',
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data} '
        '{samplestr} {wildcards.n_hidden}'

rule visualize_estimation_results:
    input:
        script='visualize_results.py',
        estimation=rules.collect_estimation_results.output.result
    output:
        plots=expand(os.path.join(
            'figures',
            '__'.join(['{{model}}', '{{data}}', '{{n_hidden}}'])
            + '__{plot}.pdf'
        ), plot=['waterfall', 'embedding', 'fit'])
    wildcard_constraints:
        model='[\w_]+',
        data='[\w_]+',
        optimzer='[\w-]+',
        n_hidden='[0-9]+',
        job='[0-9]+',
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data} '
        '{samplestr} {wildcards.n_hidden}'

rule collect_estimation:
    input:
         expand(
             rules.collect_estimation_results.output.result,
             model=PATHWAYS, data=DATASETS, n_hidden=HIDDEN_LAYERS,
         )

rule visualize_estimation:
    input:
         expand(
             rules.visualize_estimation_results.output.plots,
             model=PATHWAYS, data=DATASETS, n_hidden=HIDDEN_LAYERS,
         )