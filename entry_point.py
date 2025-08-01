import os
from graph_mirna import main_train
from graph_mirna import main_test_trino

# scope could be 'training' or 'inference'
TASK_SCOPE = os.getenv('TASK_SCOPE')


if TASK_SCOPE == 'training':
    main_train.execute_training()
else:
    main_test_trino.main()
