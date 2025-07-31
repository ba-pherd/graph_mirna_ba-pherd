import os
from graph_mirna import main_train
from graph_mirna import main_test_trino

# scope could be 'training' or 'inference'
scope = os.getenv('SCOPE', 'training')


if scope == 'training':
    main_train.execute_training()
else:
    main_test_trino.main()