from ivory.utils.config_section import ConfigSection

Pipeline = ConfigSection(
    plugins=["museek.plugin.test.test_plugin", "museek.plugin.test.test_plugin"],
)

DummyPlugin = ConfigSection(testvar=1)
