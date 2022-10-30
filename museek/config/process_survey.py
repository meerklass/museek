Pipeline = dict(
    plugins=[
        'museek.plugins.sanity_check_observation_plugin',
        'museek.plugins.sanity_check_dish_plugin'
    ]
)

SanityCheckObservationPlugin = dict(
    first_parameter=1.0  # description
)

SanityCheckDishPlugin = dict(
    second_parameter=2.0  # description
)
