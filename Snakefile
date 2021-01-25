import os

from mEncoder.training import trace_path, TRACE_FILE_TEMPLATE

HIDDEN_LAYERS = [2]
PATHWAYS = ['FLT3_MAPK_AKT_STAT']
DATASETS = ['synthetic']
OPTIMIZERS = ['fides']

STARTS = [str(i) for i in range(int(config["num_starts"]))]

localrules: process_data

rule process_data:
    input:
        script='process_data.py',
        data_code=os.path.join('mEncoder', 'generate_data.py'),
        enc_code=os.path.join('mEncoder', 'encoder.py'),
        bounds=os.path.join('mEncoder', '__init__.py'),
        model_code=os.path.join('mEncoder', 'mechanistic_model.py'),
        pathway=os.path.join('pathways', 'pw_FLT3_MAPK_AKT_STAT.py')
    output:
        datafile=os.path.join('data', '{data}__{model}.csv'),
        datafigure=os.path.join('data', '{data}__{model}.pdf')
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
        data=os.path.join('data', '{data}__{model}.csv')
    output:
        model=os.path.join('amici_models', '{model}_{data}__{model}_petab',
                           '{model}', '{model}.py'),
    wildcard_constraints:
        model='[\w_]+',
        data='[\w]+',
    shell:
        'python3 {input.script} {wildcards.data} {wildcards.model}'


rule pretrain_per_sample:
    input:
        script='pretrain_per_sample.py',
        pretraining_code=os.path.join('mEncoder', 'pretraining.py'),
        model_code=os.path.join('mEncoder', 'mechanistic_model.py'),
        model=rules.compile_mechanistic_model.output.model,
    output:
        pretraining=os.path.join(
            'pretraining', '{model}__{data}__{model}.txt'
        )
    wildcard_constraints:
        model='[\w_]+',
        data='[\w]+',
        sample='[\w_]+0',
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data}'


rule pretrain_cross_sample:
    input:
        script='pretrain_cross_samples.py',
        pretraining_code=os.path.join('mEncoder', 'pretraining.py'),
        model_code=os.path.join('mEncoder', 'mechanistic_model.py'),
        pretrain_per_sample=rules.pretrain_per_sample.output.pretraining,
        model=rules.compile_mechanistic_model.output.model,
    output:
        pretraining=os.path.join(
            'pretraining', '{model}__{data}__{model}__input.csv'
        )
    wildcard_constraints:
        model='[\w_]+',
        data='[\w]+',
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data}'

rule pretrain_encoder_inflate:
    input:
        script='pretrain_encoder_inflate.py',
        pretraining_code=os.path.join('mEncoder', 'pretraining.py'),
        model_code=os.path.join('mEncoder', 'encoder.py'),
        pretrain_cross_sample=rules.pretrain_cross_sample.output.pretraining
    output:
        pretraining=os.path.join(
            'pretraining',
            '{model}__{data}__{model}__{n_hidden}__decoder_inflate.csv'
        )
    wildcard_constraints:
        model='[\w_]+',
        data='[\w]+',
        n_hidden='[0-9]+'
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data} '
        '{wildcards.n_hidden}'


rule estimate_parameters:
    input:
        script='run_estimation.py',
        encoder_code=os.path.join('mEncoder', 'encoder.py'),
        training_code=os.path.join('mEncoder', 'training.py'),
        autoencoder_code=os.path.join('mEncoder', 'autoencoder.py'),
        dataset=rules.process_data.output.datafile,
        pretrain_encoder=rules.pretrain_encoder_inflate.output.pretraining,
        model=rules.compile_mechanistic_model.output.model,
    output:
        result=os.path.join('results', '{model}', '{data}',
                            '{optimizer}__{n_hidden}__{job}.pickle'),
        trace=os.path.join(
            trace_path,
            TRACE_FILE_TEMPLATE.format(pathway='{model}',
                                       data='{data}__{model}',
                                       optimizer='{optimizer}',
                                       n_hidden='{n_hidden}',
                                       job='{job}').replace('{id}', '0')
         )
    wildcard_constraints:
        model='[\w_]+',
        data='[\w]+',
        optimzer='[\w-]+',
        n_hidden='[0-9]+',
        job='[0-9]+',
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data} '
        '{wildcards.optimizer} {wildcards.n_hidden} {wildcards.job}'

rule collect_estimation_results:
    input:
        script='collect_estimation.py',
        trace=expand(os.path.join(
            trace_path,
            TRACE_FILE_TEMPLATE.format(pathway='{{model}}',
                                       data='{{data}}__{{model}}',
                                       optimizer='{{optimizer}}',
                                       n_hidden='{{n_hidden}}',
                                       job='{job}').replace('{id}', '0')
         ), job=STARTS)
    output:
        result=os.path.join('results', '{model}', '{data}',
                            '{optimizer}__{n_hidden}__full.pickle'),
    wildcard_constraints:
        model='[\w_]+',
        data='[\w_]+',
        optimzer='[\w-]+',
        n_hidden='[0-9]+',
        job='[0-9]+',
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data} '
        '{wildcards.n_hidden} {wildcards.optimizer}'

rule visualize_estimation_results:
    input:
        script='visualize_results.py',
        estimation=rules.collect_estimation_results.output.result
    output:
        plots=expand(os.path.join(
            'figures',
            '__'.join(['{{model}}', '{{data}}', '{{n_hidden}}',
                       '{{optimizer}}']) + '__{plot}.pdf'
        ), plot=['waterfall', 'optimizer_trace', 'embedding', 'fit',
                 'optimizer_convergence'])
    wildcard_constraints:
        model='[\w_]+',
        data='[\w_]+',
        optimzer='[\w-]+',
        n_hidden='[0-9]+',
        job='[0-9]+',
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data} '
        '{wildcards.n_hidden} {wildcards.optimizer}'

rule collect_estimation:
    input:
         expand(
             rules.collect_estimation_results.output.result,
             model=PATHWAYS, data=DATASETS, optimizer=OPTIMIZERS,
             n_hidden=HIDDEN_LAYERS,
         )

rule visualize_estimation:
    input:
         expand(
             rules.visualize_estimation_results.output.plots,
             model=PATHWAYS, data=DATASETS, optimizer=OPTIMIZERS,
             n_hidden=HIDDEN_LAYERS,
         )