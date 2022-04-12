from sklearn import svm
from sklearn import tree
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report, mean_squared_error
import xlrd
from itertools import combinations
import xlwt
from tqdm import tqdm
from statistics import mean
import multiprocessing


def load_statistics_feature():
    statistics_feature_path = 'feature/statistics_view.xls'
    return load_feature_in_excel(statistics_feature_path, range(1, 21))


def load_asr_feature():
    asr_feature_path = 'feature/asr_view.xls'
    return load_feature_in_excel(asr_feature_path, range(1, 4))


def load_group_statistics_feature():
    group_statistics_feature_path = 'feature/statistics_view_group.xls'
    group_num = 7
    group_feature_list = []
    for group_idx in range(group_num):
        group_feature_list.append(
            load_feature_in_excel(group_statistics_feature_path, range(1, 21), sheet_index=group_idx))
    return group_feature_list


def load_feature_in_excel(feature_path, range_list, sheet_index=0):
    workbook = xlrd.open_workbook(feature_path)
    worksheet = workbook.sheet_by_index(sheet_index)
    first_col = []
    for row in range(0, worksheet.nrows):
        v = worksheet.cell_value(row, 0)
        first_col.append(v)

    def get_xlsx_col_range_contents_single(col_range):
        elm_to_get = {}
        for row in range(1, worksheet.nrows):
            value_list = []
            for col_num in col_range:
                v = worksheet.cell_value(row, col_num)
                if str(v) != '':
                    value_list.append(v)
            if value_list:
                elm_to_get[first_col[row]] = value_list
        print(elm_to_get)
        return elm_to_get

    return get_xlsx_col_range_contents_single(range_list)


def load_label(regression=False):
    label_path = 'label/statistics_label.xlsx'
    workbook = xlrd.open_workbook(label_path)
    worksheet = workbook.sheet_by_index(0)
    first_col = []
    for row in range(0, worksheet.nrows):
        v = worksheet.cell_value(row, 0)
        first_col.append(v)

    def get_xlsx_col_contents_single(col_num, regression):
        elm_to_get = {}
        for row in range(1, worksheet.nrows):
            v = worksheet.cell_value(row, col_num)
            if v:
                if regression:
                    elm_to_get[first_col[row][3:]] = v
                else:
                    # MoCA threshold 26
                    elm_to_get[first_col[row][3:]] = 0 if v >= 26 else 1
        print(elm_to_get)
        return elm_to_get

    return get_xlsx_col_contents_single(3, regression)


def load_data_and_label(regression=False):
    label_dict = load_label(regression=regression)
    statistics_feature_dict = load_statistics_feature()
    group_statistics_feature_dict_list = load_group_statistics_feature()
    asr_feature_dict = load_asr_feature()

    feature_name_list = ['Overall', 'Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5',
                         'Command 16', 'Command 21', 'ASR']

    feature_list = [[] for _ in range(9)]
    label_list = []
    for key in statistics_feature_dict.keys():
        label_list.append(label_dict[key])
        feature_list[0].append(statistics_feature_dict[key])
        for group_idx, group_statistics_feature_dict in enumerate(group_statistics_feature_dict_list):
            feature_list[group_idx + 1].append(group_statistics_feature_dict[key])
        feature_list[8].append(asr_feature_dict[key])

    feature_list = [np.array(feature) for feature in feature_list]
    feature_dict = dict(zip(feature_name_list, feature_list))

    return feature_dict, np.array(label_list)


def train_and_predict_svm(x_train, y_train, x_test):
    clf = svm.SVC(random_state=1)
    clf.fit(x_train, y_train)
    y_test_pred = clf.predict(x_test)
    return y_test_pred


def train_and_predict_decision_tree(x_train, y_train, x_test):
    clf = tree.DecisionTreeClassifier(random_state=1)
    clf.fit(x_train, y_train)
    y_test_pred = clf.predict(x_test)
    return y_test_pred


def train_and_predict_random_forest(x_train, y_train, x_test):
    clf = RandomForestClassifier(n_estimators=10, random_state=1)
    clf.fit(x_train, y_train)
    y_test_pred = clf.predict(x_test)
    return y_test_pred


def train_and_predict_1nn(x_train, y_train, x_test):
    clf = MLPClassifier(hidden_layer_sizes=(5,), random_state=1, max_iter=10000, learning_rate_init=0.0001)
    clf.fit(x_train, y_train)
    y_test_pred = clf.predict(x_test)
    return y_test_pred


def train_and_predict_2nn(x_train, y_train, x_test):
    clf = MLPClassifier(hidden_layer_sizes=(10, 5), random_state=1, max_iter=10000, learning_rate_init=0.0001)
    clf.fit(x_train, y_train)
    y_test_pred = clf.predict(x_test)
    return y_test_pred


def train_and_predict_svm_regression(x_train, y_train, x_test):
    regr = svm.SVR()
    regr.fit(x_train, y_train)
    y_test_pred = regr.predict(x_test)
    return y_test_pred


def train_and_predict_decision_tree_regression(x_train, y_train, x_test):
    clf = tree.DecisionTreeRegressor(random_state=1)
    clf.fit(x_train, y_train)
    y_test_pred = clf.predict(x_test)
    return y_test_pred


def train_and_predict_random_forest_regression(x_train, y_train, x_test):
    clf = RandomForestRegressor(n_estimators=10, random_state=1)
    clf.fit(x_train, y_train)
    y_test_pred = clf.predict(x_test)
    return y_test_pred


def train_and_predict_1nn_regression(x_train, y_train, x_test):
    clf = MLPRegressor(hidden_layer_sizes=(5,), random_state=1, max_iter=10000, learning_rate_init=0.001)
    clf.fit(x_train, y_train)
    y_test_pred = clf.predict(x_test)
    return y_test_pred


def train_and_predict_2nn_regression(x_train, y_train, x_test):
    clf = MLPRegressor(hidden_layer_sizes=(10, 5), random_state=1, max_iter=10000, learning_rate_init=0.001)
    clf.fit(x_train, y_train)
    y_test_pred = clf.predict(x_test)
    return y_test_pred


def main_train_loop(data, label, classifier_train_and_predict_function, visualize=True, regression=False):
    loo = LeaveOneOut()
    # loo.get_n_splits(data)

    y_test_list = []
    y_test_pred_list = []

    for train_index, test_index in loo.split(data):
        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        y_test_pred = classifier_train_and_predict_function(x_train, y_train, x_test)
        y_test_list.append(y_test[0])
        y_test_pred_list.append(y_test_pred[0])
        # print(x_train, x_test, y_train, y_test)
    if visualize:
        if not regression:
            print(classification_report(y_test_list, y_test_pred_list, digits=4))
        else:
            print(mean_squared_error(y_test_list, y_test_pred_list, squared=False))
    if not regression:
        report_dict = classification_report(y_test_list, y_test_pred_list, digits=4, output_dict=True)
    else:
        report_dict = dict()
        report_dict['rmse'] = mean_squared_error(y_test_list, y_test_pred_list, squared=False)
    return report_dict


def main_train_loop_vote(data_list, label, classifier_train_and_predict_function, visualize=True, regression=False):
    loo = LeaveOneOut()
    # loo.get_n_splits(data)

    y_test_list = []
    y_test_pred_list = []

    for train_index, test_index in loo.split(data_list[0]):
        y_test_pred_vote_list = []
        y_test_label = None
        for data in data_list:
            # print("TRAIN:", train_index, "TEST:", test_index)
            x_train, x_test = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]
            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
            y_test_pred = classifier_train_and_predict_function(x_train, y_train, x_test)
            y_test_pred_vote_list.append(y_test_pred[0])
            if y_test_label:
                assert y_test_label == y_test[0]
            y_test_label = y_test[0]

        y_test_list.append(y_test_label)
        if not regression:
            y_test_pred_list.append(max(set(y_test_pred_vote_list), key=y_test_pred_vote_list.count))
        else:
            y_test_pred_list.append(mean(y_test_pred_vote_list))
        # print(x_train, x_test, y_train, y_test)
    if visualize:
        if not regression:
            print(classification_report(y_test_list, y_test_pred_list, digits=4))
        else:
            print(mean_squared_error(y_test_list, y_test_pred_list, squared=False))
    if not regression:
        report_dict = classification_report(y_test_list, y_test_pred_list, digits=4, output_dict=True)
    else:
        report_dict = dict()
        report_dict['rmse'] = mean_squared_error(y_test_list, y_test_pred_list, squared=False)
    return report_dict


def main_train_loop_wrapper(paras):
    comb, data_comb, label, classifier_train_and_predict_function, regression = paras
    report_dict = main_train_loop(data_comb, label, classifier_train_and_predict_function,
                                  visualize=False, regression=regression)
    return comb, report_dict


def run_classifier_report(data_dict, label, classifier_train_and_predict_function, regression=False,
                          use_multi_thread=True):
    data_name_list = list(data_dict.keys())
    data_list = list(data_dict.values())
    data_report_dict = dict()
    for data_name, data in data_dict.items():
        print('Feature set: ', data_name)
        report_dict = main_train_loop(data, label, classifier_train_and_predict_function, regression=regression)
        data_report_dict[data_name] = report_dict

    data_name = 'Separate training vote'
    print('Feature set: ', data_name)
    report_dict = main_train_loop_vote(data_list, label, classifier_train_and_predict_function, regression=regression)
    data_report_dict[data_name] = report_dict

    data_name = 'Fusion'
    print('Feature set: ', data_name)
    data = np.concatenate(data_list, axis=-1)
    report_dict = main_train_loop(data, label, classifier_train_and_predict_function, regression=regression)
    data_report_dict[data_name] = report_dict

    extra_report_dict = dict()

    combinations_list = []
    for i in range(1, len(data_dict) + 1):
        combinations_list.extend(list(combinations(range(len(data_dict)), i)))
    comb_report_dict = dict()
    if not use_multi_thread:
        for comb in tqdm(combinations_list):
            data_comb = [data_list[comb_idx] for comb_idx in comb]
            data_comb = np.concatenate(data_comb, axis=-1)
            report_dict = main_train_loop(data_comb, label, classifier_train_and_predict_function,
                                          visualize=False, regression=regression)
            comb_report_dict[comb] = report_dict
    else:
        data_comb_list = []
        for comb in tqdm(combinations_list):
            data_comb = [data_list[comb_idx] for comb_idx in comb]
            data_comb = np.concatenate(data_comb, axis=-1)
            data_comb_list.append((comb, data_comb, label, classifier_train_and_predict_function, regression))

        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            results = p.map(main_train_loop_wrapper, data_comb_list)
            p.close()
            p.join()
        for comb, report_dict in results:
            comb_report_dict[comb] = report_dict
    if not regression:
        sorted_comb_keys_by_metrics = sorted(comb_report_dict, key=lambda t: comb_report_dict[t]['accuracy'],
                                             reverse=True)
    else:
        sorted_comb_keys_by_metrics = sorted(comb_report_dict, key=lambda t: comb_report_dict[t]['rmse'],
                                             reverse=False)

    for i in range(len(sorted_comb_keys_by_metrics)):
        comb_key = sorted_comb_keys_by_metrics[i]
        data_name = ''.join([data_name_list[comb_idx] + '/' for comb_idx in comb_key])
        # print('Feature set: ', data_name)
        # print(comb_report_dict[comb_key])
        extra_report_dict[data_name] = comb_report_dict[comb_key]

    return data_report_dict, extra_report_dict


def write_xls(label_dict, extra_label_dict, regression=False):
    wb = xlwt.Workbook()
    ws = wb.add_sheet('Summary')

    classifier_name_list = list(label_dict.keys())

    for idx, label_name in enumerate(classifier_name_list):
        ws.write(0, idx + 1, label_name)

    current_row_idx = 0
    for col_idx, (classifier_name, data_report_dict) in enumerate(label_dict.items()):
        for row_idx, (data_name, report_dict) in enumerate(data_report_dict.items()):
            if col_idx == 0:
                ws.write(row_idx + 1, 0, data_name)
            if not regression:
                ws.write(row_idx + 1, col_idx + 1, report_dict['accuracy'])
            else:
                ws.write(row_idx + 1, col_idx + 1, report_dict['rmse'])
            current_row_idx = row_idx + 3

    top_k_data_name_list = set()
    for classifier_name in classifier_name_list:
        data_report_dict = extra_label_dict[classifier_name]
        data_report_dict_keys = list(data_report_dict.keys())
        for top_k_idx in range(2):
            top_k_data_name_list.add(data_report_dict_keys[top_k_idx])
    top_k_data_name_list = sorted(list(top_k_data_name_list))
    for row_idx, top_k_data_name in enumerate(top_k_data_name_list):
        ws.write(current_row_idx + row_idx + 1, 0, top_k_data_name)
        for col_idx, classifier_name in enumerate(classifier_name_list):
            if not regression:
                ws.write(current_row_idx + row_idx + 1, col_idx + 1,
                         extra_label_dict[classifier_name][top_k_data_name]['accuracy'])
            else:
                ws.write(current_row_idx + row_idx + 1, col_idx + 1,
                         extra_label_dict[classifier_name][top_k_data_name]['rmse'])

    if not regression:
        metrics_list = ['accuracy', 'class 0 precision', 'class 0 recall', 'class 0 f1',
                        'class 1 precision', 'class 1 recall', 'class 1 f1', ]
    else:
        metrics_list = ['rmse', ]
    for classifier_name in classifier_name_list:
        ws = wb.add_sheet(classifier_name)
        for idx, label_name in enumerate(metrics_list):
            ws.write(0, idx + 1, label_name)
        data_report_dict = extra_label_dict[classifier_name]
        for row_idx, (data_name, report_dict) in enumerate(data_report_dict.items()):
            ws.write(row_idx + 1, 0, data_name)
            if not regression:
                ws.write(row_idx + 1, 1, report_dict['accuracy'])
                ws.write(row_idx + 1, 2, report_dict['0']['precision'])
                ws.write(row_idx + 1, 3, report_dict['0']['recall'])
                ws.write(row_idx + 1, 4, report_dict['0']['f1-score'])
                ws.write(row_idx + 1, 5, report_dict['1']['precision'])
                ws.write(row_idx + 1, 6, report_dict['1']['recall'])
                ws.write(row_idx + 1, 7, report_dict['1']['f1-score'])
            else:
                ws.write(row_idx + 1, 1, report_dict['rmse'])

    if not regression:
        wb.save('classification_view.xls')
    else:
        wb.save('regression_view.xls')


def run_classification():
    data_dict, label = load_data_and_label()
    svm_data_report_dict, svm_extra_report_dict = run_classifier_report(data_dict, label, train_and_predict_svm)
    decision_tree_data_report_dict, decision_tree_extra_report_dict = \
        run_classifier_report(data_dict, label, train_and_predict_decision_tree)
    nn1_data_report_dict, nn1_extra_report_dict = run_classifier_report(data_dict, label, train_and_predict_1nn)
    nn2_data_report_dict, nn2_extra_report_dict = run_classifier_report(data_dict, label, train_and_predict_2nn)
    random_forest_data_report_dict, random_forest_extra_report_dict = \
        run_classifier_report(data_dict, label, train_and_predict_random_forest)

    total_data_report_dict = {'svm': svm_data_report_dict, 'decision tree': decision_tree_data_report_dict,
                              '1nn': nn1_data_report_dict, '2nn': nn2_data_report_dict,
                              'random forest': random_forest_data_report_dict,
                              }
    total_extra_data_report_dict = {'svm': svm_extra_report_dict, 'decision tree': decision_tree_extra_report_dict,
                                    '1nn': nn1_extra_report_dict, '2nn': nn2_extra_report_dict,
                                    'random forest': random_forest_extra_report_dict,
                                    }

    write_xls(total_data_report_dict, total_extra_data_report_dict)


def run_regression():
    data_dict, label = load_data_and_label(regression=True)
    svm_data_report_dict, svm_extra_report_dict = \
        run_classifier_report(data_dict, label, train_and_predict_svm_regression, regression=True)
    decision_tree_data_report_dict, decision_tree_extra_report_dict = \
        run_classifier_report(data_dict, label, train_and_predict_decision_tree_regression, regression=True)
    nn1_data_report_dict, nn1_extra_report_dict = \
        run_classifier_report(data_dict, label, train_and_predict_1nn_regression, regression=True)
    nn2_data_report_dict, nn2_extra_report_dict = \
        run_classifier_report(data_dict, label, train_and_predict_2nn_regression, regression=True)
    random_forest_data_report_dict, random_forest_extra_report_dict = \
        run_classifier_report(data_dict, label, train_and_predict_random_forest_regression, regression=True)

    total_data_report_dict = {'svm': svm_data_report_dict, 'decision tree': decision_tree_data_report_dict,
                              '1nn': nn1_data_report_dict, '2nn': nn2_data_report_dict,
                              'random forest': random_forest_data_report_dict,
                              }
    total_extra_data_report_dict = {'svm': svm_extra_report_dict, 'decision tree': decision_tree_extra_report_dict,
                                    '1nn': nn1_extra_report_dict, '2nn': nn2_extra_report_dict,
                                    'random forest': random_forest_extra_report_dict,
                                    }

    write_xls(total_data_report_dict, total_extra_data_report_dict, regression=True)


def main():
    run_classification()
    run_regression()


if __name__ == '__main__':
    main()
