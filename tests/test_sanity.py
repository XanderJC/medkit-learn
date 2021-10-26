import medkit as mk


def test_sanity() -> None:
    synthetic_dataset = mk.batch_generate(
        domain="Ward",
        environment="CRN",
        policy="LSTM",
        size=200,
        test_size=20,
        max_length=10,
        scale=True,
    )

    static_train, observations_train, actions_train = synthetic_dataset["training"]
    static_test, observations_test, actions_test = synthetic_dataset["testing"]

    for data in [static_train, observations_train, actions_train]:
        assert data.shape[0] == 200

    for data in [static_test, observations_test, actions_test]:
        assert data.shape[0] == 20
