# -*- coding: utf-8 -*-
import os
import logging
import psycopg2
import sys
import pandas as pd
import statistics
import requests
from datetime import timedelta, datetime


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def create_db_conn(database, user, password, host, port=None):
    try:
        conn = psycopg2.connect(database=database,
                                user=user,
                                password=password,
                                host=host,
                                port=port)
    except:
        logger.exception("ERROR: Could not connect to Postgres instance.")
        sys.exit()
    
    logger.info("SUCCESS: Connect to RDS Postgres instance.")
    return conn


def insert_into_db(conn, table_name, columns, records, return_cols=None):
    if return_cols:
        returning = ' RETURNING ' + ', '.join(return_cols)
    else:
        returning = ''

    with conn.cursor() as cur:
        insert_stmt = 'INSERT INTO {table_name} ({columns}) VALUES {str_records}{returning};'.format(
            table_name=table_name,
            columns=', '.join(columns),
            str_records=', '.join(map(str, records)),
            returning=returning)
        cur.execute(insert_stmt)
        conn.commit()
        logger.info('insert into database')
        if returning:
            return cur.fetchall()


def update_db(conn, table_name, customer_overview_id, update_data):
    with conn.cursor() as cur:
        update_stmt = 'UPDATE {table_name} SET {update} WHERE id = {customer_overview_id};'.format(
            table_name=table_name,
            update=', '.join([f'{k}={v}' for k, v in update_data.iteritems()]),
            customer_overview_id=customer_overview_id)
        cur.execute(update_stmt)
        conn.commit()
    logger.info('update database')


def get_data(conn, cdp_id):
    data = pd.read_sql(
        "SELECT DISTINCT * FROM imported_orders WHERE cdp_id = {cdp_id} AND paid_at >= '{since_date}'".format(
            cdp_id=cdp_id,
            since_date=(datetime.now() - timedelta(365)).strftime('%Y-%m-%d')),
            conn)
    data = data[['order_id', 'buyer_id', 'total_amount', 'paid_at', 'order_status', 'return_status']]
    return data


def data_cleansing(data, combine_orders=False):
    if combine_orders:
        total_amount = data.groupby('order_id').agg({'total_amount': sum})
        data = data.drop_duplicates(subset='order_id').drop('total_amount', axis=1)
        data = data.merge(total_amount, on='order_id')

    data['order_status'] = data['order_status'].astype(str)
    data = data[data['order_status'] == '1']

    try:
        data['paid_at'] = pd.to_datetime(data['paid_at'], format='%Y-%m-%d %H:%M')
    except:
        data['paid_at'] = pd.to_datetime(data['paid_at'], format='%d/%m/%Y %H:%M')

    return data


def customer_active_index(data):
    logger.info('customer_active_index start')
    end_date = data['paid_at'].max() + timedelta(days=1)
    result = pd.DataFrame(columns=['buyer_id', 'cai'])
    group_data = data.groupby('buyer_id')
    for name, group in group_data:
        if len(group) > 1:
            group = group.sort_values('paid_at').reset_index(drop=True)
            days_since_first_trans = end_date - group.loc[0, 'paid_at']
            avg_trans_period = days_since_first_trans / len(group)
            group['time_to_next_trans'] = group['paid_at'].shift(-1) - group['paid_at']
            group.tail(1)['time_to_next_trans'] = end_date - group.tail(1)['paid_at']
            weighted_sum_trans_period = 0
            weight = 0
            weight_sum = 0
            for i, row in group.iterrows():
                weight = i + 1
                weighted_sum_trans_period += weight * row['time_to_next_trans'].total_seconds()
                weight_sum += weight
            weighted_avg_trans_period = timedelta(seconds=weighted_sum_trans_period) / weight_sum
            cai = (avg_trans_period - weighted_avg_trans_period) / avg_trans_period
            if cai > 0:
                cai_tag = 1  # active customer
            else:
                cai_tag = -1  # inactive customer
        else:
            cai_tag = 0  # Can not calculate CAI due to lack of order data.
        result = result.append({'buyer_id': name, 'cai': cai_tag}, ignore_index=True)
    logger.info('customer_active_index done')
    return result.set_index('buyer_id')


def rfm_tagging(data):
    logger.info('rfm_tagging start')
    rfm_data = data.groupby('buyer_id').agg({'paid_at': [max, len], 'total_amount': sum})
    rfm_data.columns = ['recency', 'frequency', 'monetary']
    rfm_data['tag'] = ''
    high_r = rfm_data['recency'] > rfm_data['recency'].mean()
    high_f = rfm_data['frequency'] > rfm_data['frequency'].mean()
    high_m = rfm_data['monetary'] > rfm_data['monetary'].mean()
    rfm_data.loc[high_r & high_f & high_m, 'tag'] = 'h_rfm'
    rfm_data.loc[high_r & high_f & ~high_m, 'tag'] = 'h_rf'
    rfm_data.loc[high_r & ~high_f & high_m, 'tag'] = 'h_rm'
    rfm_data.loc[~high_r & high_f & high_m, 'tag'] = 'h_fm'
    rfm_data.loc[high_r & ~high_f & ~high_m, 'tag'] = 'h_r'
    rfm_data.loc[~high_r & high_f & ~high_m, 'tag'] = 'h_f'
    rfm_data.loc[~high_r & ~high_f & high_m, 'tag'] = 'h_m'
    rfm_data.loc[~high_r & ~high_f & ~high_m, 'tag'] = 'h_none'
    rfm_data.to_records(index=False).tolist()
    logger.info('rfm_tagging done')
    return rfm_data


def rfm_overview(data):
    rfm_summary = data.groupby('tag').agg({'frequency': 'count', 'monetary': sum})
    rfm_summary.columns = ['customers', 'total_amount']
    rfm_summary['customers_pct'] = round(rfm_summary['customers'] / rfm_summary['customers'].sum(), 4)
    rfm_summary['total_amount_pct'] = round(rfm_summary['total_amount'] / rfm_summary['total_amount'].sum(), 4)
    rfm_summary = rfm_summary.stack().to_frame().T
    rfm_summary.columns = ['{}_{}'.format(*c) for c in rfm_summary.columns]
    return rfm_summary


def create_overview(cdp_id, db_conn):
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    updated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    table_name = 'customer_overviews'
    cols = ['cdp_id', 'created_at', 'updated_at']
    records = [(cdp_id, created_at, updated_at)]
    return_cols = ['id']
    customer_overview_id = insert_into_db(conn=db_conn,
                                    table_name=table_name,
                                    columns=cols,
                                    records=records,
                                    return_cols=return_cols)
    return customer_overview_id


def customer_tagging(cdp_id, customer_overview_id, data, db_conn):
    logger.info('customer_tagging start')
    tagged_data = rfm_tagging(data)
    cai_data = customer_active_index(data)
    insert_data = tagged_data.join(cai_data).reset_index()
    insert_data['cdp_id'] = cdp_id
    insert_data['customer_overview_id'] = customer_overview_id
    insert_data['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    insert_data['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    table_name = 'rfm_tag_temps'
    tag_cols = [
        'cdp_id', 'customer_overview_id', 'buyer_id',
        'tag', 'cai', 'created_at', 'updated_at'
    ]
    tag_records = insert_data[tag_cols].to_records(index=False)
    
    insert_into_db(conn=db_conn,
                   table_name=table_name,
                   columns=tag_cols,
                   records=tag_records)

    result_data = insert_data.rename(columns={
        'recency': 'last_purchased_at',
        'frequency': 'purchase_counts',
        'monetary': 'total_purchase_amount'
    })

    table_name = 'rfm_results'
    result_cols = [
        'cdp_id', 'customer_overview_id', 'buyer_id', 'last_purchased_at',
        'purchase_counts', 'total_purchase_amount', 'created_at', 'updated_at'
    ]
    result_records = result_data[result_cols].to_records(index=False)

    insert_into_db(conn=db_conn,
                   table_name=table_name,
                   columns=result_cols,
                   records=result_records)

    logger.info('customer_tagging done')

    return insert_data


def customer_overview(customer_overview_id, tagged_data, db_conn):
    logger.info('customer_overview start')
    overview = rfm_overview(tagged_data)
    
    table_name = 'customer_overviews'
    update_db(conn=db_conn,
              table_name=table_name,
              customer_overview_id=customer_overview_id,
              update_data=overview.loc[0])

    logger.info('customer_overview done')


def rfm_ready(cdp_id, customer_overview_id):
    logger.info('rfm_ready start')
    
    api_url = os.environ.get('oceanRfmReadyApi')
    logger.info(f'URL: {api_url}, cdp_id: {cdp_id}, overview_id: {customer_overview_id}')

    params = {'cdp_id': cdp_id,
              'customer_overview_id': customer_overview_id}
    try:
        requests.post(api_url, params=params)
    except:
        requests.post(api_url, params=params)

    logger.info('rfm_ready done')


# main
def rfm_handler(event, context):
    # create db connection
    database = os.environ.get('databaseName')
    user = os.environ.get('databaseUser')
    password = os.environ.get('databasePassword')
    host = os.environ.get('databaseHost')
    port = os.environ.get('databasePort')
    conn = create_db_conn(database, user, password, host, port)

    try:
        # get query parameters
        cdp_id = event['queryStringParameters']['cdp_id']
    except:
        logger.exception("ERROR: Missing query parameters.")
        return {"statusCode": "400"}

    try:
        # load customer data
        data = get_data(conn, cdp_id)
        data = data_cleansing(data, combine_orders=True)
    except:
        logger.exception("ERROR: Data prepared error.")
        return {"statusCode": "500"}

    try:
        # create new record in customer_overview an get id
        customer_overview_id = create_overview(cdp_id, conn)[0][0]
        # rfm
        tagged_data = customer_tagging(cdp_id, customer_overview_id, data, conn)
        customer_overview(customer_overview_id, tagged_data, conn)
        # send a request to rfm_ready_api
    except:
        logger.exception("ERROR: RFM analysis error.")
        return {"statusCode": "500"}

    try:
        rfm_ready(cdp_id, customer_overview_id)
        return {"statusCode": "200"}
    except:
        logger.exception("ERROR: RFM ready api error.")
        return {"statusCode": "500"}
    finally:
        conn.close()
