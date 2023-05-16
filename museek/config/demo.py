from ivory.utils.config_section import ConfigSection

Pipeline = ConfigSection(
    plugins=[
        'museek.plugin.demo.demo_load_plugin',
        'museek.plugin.demo.demo_flip_plugin',
        'museek.plugin.demo.demo_plot_plugin',
        'museek.plugin.demo.demo_joblib_plugin'
    ],
)

DemoLoadPlugin = ConfigSection(
    url='https://cdn.openai.com/dall-e-2/demos/text2im/astronaut/horse/photo/9.jpg',
    context_file_name='context.pickle'
)

DemoPlotPlugin = ConfigSection(
    do_show=False,
    do_save=True
)

DemoFlipPlugin = ConfigSection(
    do_flip_right_left=True,
    do_flip_top_bottom=True
)

DemoJoblibPlugin = ConfigSection(
    n_iter=10,
    n_jobs=2,
    verbose=0
)
