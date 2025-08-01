import os
# scope could be 'training' or 'inference'
TASK_SCOPE = os.getenv('TASK_SCOPE', 'training')

if TASK_SCOPE == 'training':
    from graph_mirna import main_train
    print('Starting training...')
    main_train.execute_training()
else:
    from graph_mirna import main_test_trino
    print('Starting inference...')
    main_test_trino.main()
