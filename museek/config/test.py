from ivory.utils.config_section import ConfigSection

Pipeline = ConfigSection(
    plugins=[
        'museek.plugin.test.test_plugin',
        'museek.plugin.test.test_plugin'
    ],
)

TestPlugin = ConfigSection(
    testvar=1
)

TestPlugin2 = ConfigSection(
    testvar=2
)

