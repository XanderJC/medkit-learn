from medkit.scenario import scenario

def generate_data(domain        = 'ICU',
                    environment = 'RNN',
                    policy      = 'RNN',
                    size        = 100,
                    test_split  = False,
                    **kwargs):
    '''
    Base API function for generating a batch dataset 
    '''
    scene = scenario(domain,environment,policy)

    data = scene.batch_generate(**kwargs)

    return data


if __name__ == '__main__':

    data_total = generate_data()
    '''
    There's something dodge in scenario.batch_generate() that's causing it to get stuck occasionally
    I think - will investigate later.
    '''
    print(data_total[0].shape)
    print(data_total[1].shape)
    print(data_total[2].shape)
