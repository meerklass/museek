from ivory.utils.config_section import ConfigSection

Pipeline = ConfigSection(
    plugins=[
        'museek.plugin.sanity_check_observation_plugin',
        'museek.plugin.sanity_check_dish_plugin'
    ]
)

SanityCheckObservationPlugin = ConfigSection(
    first_parameter=1.0  # description
)

SanityCheckDishPlugin = ConfigSection(
    second_parameter=2.0  # description
)
