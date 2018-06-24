import pandas
import numpy as np
def save_submit_file(x_test, model, filename):
    test_prediction = model.predict(x_test)
    test_labels = np.argmax(test_prediction, axis = 1)
    df = pandas.DataFrame(data={"Category": test_labels}).astype(int)
    df.to_csv(filename, sep=',',index=True,  index_label='Id')