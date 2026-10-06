import datetime
import time
import random
import logging 
import uuid
import pytz
import pandas as pd
import io
import psycopg

#  Python 标准库 logging 的一次性全局配置，用来告诉日志系统：最低记录什么级别、日志长什么样。
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
# 给根 logger 设最低级别为 INFO，并规定日志格式为「时间 [级别]: 消息。
# 低于 INFO 的日志被过滤，输出形如 2026-10-06 14:30:05,123 [INFO]: 服务启动成功。
# 它只生效一次，默认输出到 stderr，通常放在程序入口调用。



SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
	timestamp timestamp,
	value1 integer,
	value2 varchar,
	value3 float
)
"""
# 创建test数据库和 dummy_metrics 表
def prep_db():
	with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
		res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
		if len(res.fetchall()) == 0:
			conn.execute("create database test;")
		with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
			conn.execute(create_table_statement)

# 计算并插入随机数据
def calculate_dummy_metrics_postgresql(curr):
	value1 = rand.randint(0, 1000)
	value2 = str(uuid.uuid4())
	value3 = rand.random()

	curr.execute(
		"insert into dummy_metrics(timestamp, value1, value2, value3) values (%s, %s, %s, %s)",
		(datetime.datetime.now(pytz.timezone('Europe/London')), value1, value2, value3)
	)

# 主函数，负责调用其他函数，计算并插入随机数据
def main():
	prep_db()
	last_send = datetime.datetime.now()
	with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
		for _ in range(100):
			with conn.cursor() as curr:
				calculate_dummy_metrics_postgresql(curr)

			new_send = datetime.datetime.now()
			seconds_elapsed = (new_send - last_send).total_seconds()
			if seconds_elapsed < SEND_TIMEOUT:
				time.sleep(SEND_TIMEOUT - seconds_elapsed)
			last_send = last_send + datetime.timedelta(seconds=10)
			logging.info("data sent")

if __name__ == "__main__":
	main()
