from plotnine import *
import matplotlib.pyplot as plt
import petab
import os

PLOTNINE_THEME = {
    'dpi': 300,
    'legend_background': element_blank(),
    'legend_key': element_blank(),
    'panel_background': element_blank(),
    'strip_background': element_blank(),
    'axis_line': element_line(size=1),
}


def plot_cross_samples(measurement_df, simulation_df, figdir, prefix):
    measurement_df = measurement_df.reset_index()
    simulation_df = simulation_df.copy()

    dyn_sim = simulation_df[
        simulation_df[petab.PREEQUILIBRATION_CONDITION_ID] !=
        simulation_df[petab.SIMULATION_CONDITION_ID]
    ]
    dyn_mes = measurement_df[
        measurement_df[petab.PREEQUILIBRATION_CONDITION_ID] !=
        measurement_df[petab.SIMULATION_CONDITION_ID]
    ]

    for df in [dyn_sim, dyn_mes]:
        df[petab.SIMULATION_CONDITION_ID] = \
            df[petab.SIMULATION_CONDITION_ID].apply(
                lambda x: x.split('__')[1]
            )

    kwargs = {
        'x': 'time',
        'color': petab.PREEQUILIBRATION_CONDITION_ID,
        'group': petab.PREEQUILIBRATION_CONDITION_ID
    }
    plot = (
        ggplot() +
        geom_line(
            data=dyn_sim,
            mapping=aes(y=petab.SIMULATION, **kwargs),
            size=1,
        )
        + geom_point(
            data=dyn_mes,
            mapping=aes(y=petab.MEASUREMENT, **kwargs),
            size=1,
        )
        + facet_grid((petab.OBSERVABLE_ID, petab.SIMULATION_CONDITION_ID),
                     scales='free_y')
        + xlab('time [min]')
        + ylab('measurement')
        + theme(**PLOTNINE_THEME)
    )

    plt.tight_layout()
    plot.save(os.path.join(figdir, f'{prefix}_fit_dynamic.pdf'))

    simulation_df[petab.MEASUREMENT] = measurement_df[petab.MEASUREMENT]
    stat = simulation_df[
        simulation_df[petab.PREEQUILIBRATION_CONDITION_ID] ==
        simulation_df[petab.SIMULATION_CONDITION_ID]
    ]

    plot = (
            ggplot(data=stat,
                   mapping=aes(y=petab.MEASUREMENT, x=petab.SIMULATION,
                               color=petab.PREEQUILIBRATION_CONDITION_ID,
                               group=petab.PREEQUILIBRATION_CONDITION_ID))
            + geom_point(size=1)
            + facet_wrap(petab.OBSERVABLE_ID,
                         scales='free_y')
            + xlab('simulation')
            + ylab('measurement')
            + theme(**PLOTNINE_THEME)
    )

    plt.tight_layout()
    plot.save(os.path.join(figdir, f'{prefix}_fit_static.pdf'))

