# Evidently 的调试脚本，核心是：用 TestSuite 和 Report 两种方式，
# 对某一天的数据做数据漂移检查，定位问题。

import datetime
import pandas as pd

from evidently import ColumnMapping
# Report + DataDriftPreset：生成详细报告，展示每个特征的漂移分数、分布图
# 分析根因   适合：人工调试
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# TestSuite + DataDriftTestPreset：生成测试套件，每个检查项返回 Success/fail
# 自动化断言、告警    适合：CI/CD、定时检查
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset

from joblib import dump, load


# Load data and model
ref_data = pd.read_parquet('../data/reference.parquet') 
current_data = pd.read_parquet('../data/green_tripdata_2022-02.parquet')

with open('../models/lin_reg_monitor_base_line.bin', 'rb') as f_in:
    model = load(f_in)

# data labeling
num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
cat_features = ["PULocationID", "DOLocationID"]

# 切出"问题数据",取 2022 年 2 月 2 日一整天的数据。这是在 Grafana 里发现异常的那一天，现在要深挖。
problematic_data = current_data.loc[(current_data.lpep_pickup_datetime >= datetime.datetime(2022,2,2,0,0)) & 
                               (current_data.lpep_pickup_datetime < datetime.datetime(2022,2,3,0,0))]


# Generate Test Suite and Report
column_mapping = ColumnMapping(
    prediction='prediction',
    numerical_features=num_features,
    categorical_features=cat_features,
    target=None
)

problematic_data['prediction'] = model.predict(problematic_data[num_features + cat_features].fillna(0))  

test_suite = TestSuite(tests = [DataDriftTestPreset()])
test_suite.run(reference_data=ref_data, current_data=problematic_data, column_mapping=column_mapping)

#test_suite.show(mode='inline')

report = Report(metrics = [DataDriftPreset()])
report.run(reference_data=ref_data, current_data=problematic_data, column_mapping=column_mapping)

#report.show(mode='inline')
