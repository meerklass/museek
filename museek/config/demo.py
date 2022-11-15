from ivory.utils.config_section import ConfigSection

Pipeline = ConfigSection(
    plugins=[
        'museek.plugins.demo_load_plugin',
        'museek.plugins.demo_flip_plugin',
        'museek.plugins.demo_plot_plugin',
        'museek.plugins.demo_store_plugin'
    ]
)

DemoLoadPlugin = ConfigSection(
    url='https://cdn.openai.com/dall-e-2/demos/text2im/astronaut/horse/photo/9.jpg'
)

DemoPlotPlugin = ConfigSection(
    do_show=True
)

DemoFlipPlugin = ConfigSection(
    do_flip_right_left=True,
    do_flip_top_bottom=True
)

DemoStorePlugin = ConfigSection(
    file_name='context.pickle'
)
