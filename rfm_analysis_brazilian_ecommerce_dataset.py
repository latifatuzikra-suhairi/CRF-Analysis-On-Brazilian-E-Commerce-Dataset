import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import pandas as pd

# read data
resp = urlopen("https://raw.githubusercontent.com/latifatuzikra-suhairi/CRF-Analysis-On-Brazilian-E-Commerce-Dataset/main/Dataset/preprocessing_dataset.zip")
zipfile = ZipFile(BytesIO(resp.read()))
extracted_file = zipfile.open(zipfile.namelist()[0])
df = pd.read_csv(extracted_file)

# create dataframe
def create_customers_by_state(df_all):
    customers_by_state = df_all.groupby(by="customer_state").agg(num_customer = ('customer_unique_id', 'nunique')).reset_index()
    customers_by_state_desc_df = customers_by_state.sort_values(by=["num_customer"], ascending=False)
    return customers_by_state_desc_df

def create_customers_by_city(df_all):
    customers_by_city = df_all.groupby(by="customer_city").agg(num_customer = ('customer_unique_id', 'nunique')).reset_index()
    customers_by_city_desc_df = customers_by_city.sort_values(by=["num_customer"], ascending=False)
    return customers_by_city_desc_df

def create_product_cat_name(df_all):
    order_by_product_cat = df_all.groupby(by="product_category_name").agg(num_order = ('order_id', 'count'))
    order_by_product_cat = order_by_product_cat.sort_values(by=['num_order'], ascending=False).reset_index()
    return order_by_product_cat

def create_payment_type(df_all):
    payment_by_order = df_all.groupby(by="payment_type").agg(num_order = ('order_id', 'nunique'))
    payment_by_order_desc = payment_by_order.sort_values(by=['num_order'], ascending=False).reset_index()
    return payment_by_order_desc

def create_num_order_by_month(df_all):
    df_all['order_purchase_timestamp'] = pd.to_datetime(df_all['order_purchase_timestamp'], format='%Y-%m-%d %H:%M:%S')
    df_all.insert(5,'month_order', df_all['order_purchase_timestamp'].dt.strftime('%m-%Y'))
    order_by_month = df_all.resample(rule='M', on='order_purchase_timestamp').agg(num_order = ('order_id','nunique'), sum_total_order = ('total_order','sum')).reset_index()
    order_by_month = order_by_month.rename(columns ={"order_purchase_timestamp":"month_order"})
    order_by_month['month_order'] = order_by_month ['month_order'].dt.strftime('%m-%Y')
    order_by_month.sort_values(by=['num_order'], ascending=False).head()
    return order_by_month

def create_rfm_df(df_all):
    df_rfm = df_all.groupby(by="customer_unique_id", as_index=False).agg(
          max_order_date = ('order_purchase_timestamp', 'max'), #ambil tanggal order terakhir pelanggan
          frequency =  ('order_id', 'nunique'),  #ambil frekuensi order pelanggan, disimpan sebagai nilai frequency
          monetary = ('total_order', 'sum') #ambil total nominal belanja pelanggan, disimpan sebagai nilai monetary
    )
    recent_date_order = df_all['order_purchase_timestamp'].dt.date.max() #ambil data order terakhir kalinya
    df_rfm['max_order_date'] = df_rfm['max_order_date'].dt.date #mengubah tipe data ke date
    df_rfm.insert(1, 'recency', df_rfm['max_order_date'].apply(lambda max_order: (recent_date_order - max_order).days)) #selisih hari date order terkahir pelanggan ybs dengan date terakhir keseluruhan pelanggan
    df_rfm.drop(columns=['max_order_date'], inplace=True)

    df_rfm['R_rank'] = df_rfm['recency'].rank(ascending=False)
    df_rfm['F_rank'] = df_rfm['frequency'].rank(ascending=True)
    df_rfm['M_rank'] = df_rfm['monetary'].rank(ascending=True)
    
    df_rfm['R_rank_norm'] = (df_rfm['R_rank']/df_rfm['R_rank'].max())*100
    df_rfm['F_rank_norm'] = (df_rfm['F_rank']/df_rfm['F_rank'].max())*100
    df_rfm['M_rank_norm'] = (df_rfm['F_rank']/df_rfm['M_rank'].max())*100
    df_rfm.drop(columns=['R_rank', 'F_rank', 'M_rank'], inplace=True)

    # weighting masing-masing parameter
    df_rfm['RFM_score'] = 0.18*df_rfm['R_rank_norm'] + 0.30 *df_rfm['F_rank_norm'] + 0.52*df_rfm['M_rank_norm']
    # ubah nilai RFM Score menjadi nilai dengan maksimal 5 dan membulatkannya hingga 2 desimal
    df_rfm['RFM_score'] = (0.05*df_rfm['RFM_score']).round(2)

    return df_rfm

def create_customer_segment(df_rfm):
    # segmentasi
    df_rfm["customer_segment"] = np.where(
    df_rfm['RFM_score'] > 4.0, "Top Customer", (np.where(
        df_rfm['RFM_score'] > 3.0, "High Value Customer",(np.where(
            df_rfm['RFM_score'] > 2.0, "Medium Value Customer", np.where(
                df_rfm['RFM_score'] > 1.5, 'Low Value Customer', 'Lost Customer')))))
    )

    df_rfm_segment = df_rfm.groupby(by="customer_segment").agg(num_cust_seg = ('customer_segment', 'count')).reset_index()
    df_rfm_segment = df_rfm_segment.sort_values(by=['num_cust_seg'], ascending=False).reset_index()
    return df_rfm_segment

# panggil dataframe
customers_by_state_df = create_customers_by_state(df)
customers_by_city_df = create_customers_by_city(df)
product_category_desc_df = create_product_cat_name(df)
product_category_asc_df = create_product_cat_name(df).sort_values(by=['num_order'], ascending=True).reset_index()
payment_type_df = create_payment_type(df)
order_by_month_df = create_num_order_by_month(df)
df_rfm = create_rfm_df(df)
rfm_segment_df = create_customer_segment(create_rfm_df(df))

#navigasi sidebar
with st.sidebar:
    selected = option_menu ("Analisis RFM untuk Segmentasi Pelanggan (Studi Kasus: Olist Store)",
                            ["Dataset Yang Digunakan",
                             "Hasil Analisis Data"],
                             default_index=0)

# halaman
if (selected == "Dataset Yang Digunakan"):
    st.title("Dataset Brazilian E-Commerce Olist Store")
    with st.container():
        st.write('''Merupakan dataset E-Commerce Brazil yang menyimpan data transaksi yang terjadi pada Olist Store. 
                 Dataset memiliki 100 ribu data transaksi dari tahun 2016 hingga 2018 yang dilakukan di beberapa pasar di Brazil. 
                 Dilengkapi dengan beberapa atribut data yang memungkinkan penggunanya melihat transaksi dari berbagai dimensi: 
                 mulai dari status pesanan, harga, pembayaran dan kinerja pengiriman hingga lokasi pelanggan, atribut produk, dan ulasan yang ditulis oleh pelanggan. 
                 Dataset ini juga menyediakan kumpulan data geolokasi yang menghubungkan kode pos Brasil dengan koordinat lintang/bujur.''')
        st.write('''\nMenggunakan Dataset Brazilian E-Commerce Olist Store, akan dilakukan analisis RFM untuk membantu Olist Store 
                 untuk menyaring pelanggan ke dalam berbagai kelompok pelanggan. Kelompok pelanggan ini selanjutnya digunakan oleh manajer untuk 
                 mengidentifikasi kelompok pelanggan mana yang dapat membuat bisnis Olist Store lebih menguntungkan.''')
        st.caption('@latifatuzikra | 2023')
if (selected == "Hasil Analisis Data") :
    st.title("Brazilian E-Commerce Olist Store Dashboard :sparkles:")

    #visualisasi 1
    st.subheader('Sebaran Pelanggan Berdasarkan Negara Asal dan Kota Asal')
    fig1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

    sns.barplot(x="customer_state", y="num_customer", data=customers_by_state_df.head(5), palette='Blues_r', ax=ax1[0])
    ax1[0].bar_label(ax1[0].containers[0])
    ax1[0].set_ylabel("Jumlah Pelanggan", fontsize=15)
    ax1[0].set_xlabel("Negara", fontsize=15)
    ax1[0].tick_params(axis='x', labelrotation=45)
    ax1[0].set_title("Berdasarkan Negara", loc="center", fontsize=18)
    ax1[0].tick_params(axis ='y', labelsize=15)

    sns.barplot(x="customer_city", y="num_customer", palette='Blues_r', data= customers_by_city_df.head(5), ax=ax1[1])
    ax1[1].bar_label(ax1[1].containers[0])
    ax1[1].set_ylabel("Jumlah Pelanggan", fontsize=15)
    ax1[1].set_xlabel("Kota", fontsize=15)
    ax1[1].tick_params(axis='x', labelrotation=45)
    ax1[1].set_title("Berdasarkan Kota", loc="center", fontsize=18)
    ax1[1].tick_params(axis='y', labelsize=15)
    st.pyplot(fig1)

    # visualisasi2
    st.subheader('5 Kategori Produk Yang Paling Banyak dan Paling Sedikit Di Order')
    fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

    sns.barplot(x="product_category_name", y="num_order", data=product_category_desc_df.head(5), palette='Blues_r', ax=ax2[0])
    ax2[0].bar_label(ax2[0].containers[0])
    ax2[0].set_ylabel("Jumlah Order", fontsize=15)
    ax2[0].set_xlabel("Kategori Produk", fontsize=15)
    ax2[0].tick_params(axis='x', labelrotation=45)
    ax2[0].set_title("Kategori Produk Yang Paling Banyak Di Order", loc="center", fontsize=18)
    ax2[0].tick_params(axis ='y', labelsize=15)

    sns.barplot(x="product_category_name", y="num_order", palette='Blues_r', data= product_category_asc_df.head(5), ax=ax2[1])
    ax2[1].bar_label(ax2[1].containers[0])
    ax2[1].set_ylabel("Jumlah Order", fontsize=15)
    ax2[1].set_xlabel("Kategori Produk", fontsize=15)
    ax2[1].tick_params(axis='x', labelrotation=45)
    ax2[1].set_title("Kategori Produk Yang Paling Sedikit Di Order", loc="center", fontsize=18)
    ax2[1].tick_params(axis='y', labelsize=15)
    st.pyplot(fig2)

    # visualisasi3
    st.subheader('Sebaran Tipe Pembayaran Yang Digunakan Pelanggan')
    fig3, ax3 = plt.subplots(figsize=(6,6))
    palette_color = sns.color_palette('Blues_r')
    ax3.pie(payment_type_df['num_order'], autopct='%1.1f%%', colors=palette_color)
    ax3.legend(payment_type_df['payment_type'], loc='upper right', bbox_to_anchor=(1.4, 1))
    st.pyplot(fig3)

    #visualisasi4
    st.subheader('Perkembangan Jumlah Order Setiap Bulannya Pada Tahun 2016 hingga 2018')
    fig4, ax4 = plt.subplots(figsize=(20,6))
    ax4 = sns.lineplot(x="month_order", y="num_order", data=order_by_month_df, marker='o', color='navy', label='Jumlah Order')
    ax4.set_ylabel("Jumlah Order", fontsize=15)
    ax4.set_xlabel("Bulan", fontsize=15)
    ax4.legend(loc='upper right')
    ax4.tick_params(axis='x', labelrotation=15)

    for line in range(0, order_by_month_df.shape[0]):
        ax4.text(order_by_month_df.month_order[line], order_by_month_df.num_order[line], order_by_month_df.num_order[line], horizontalalignment='left', verticalalignment='top', size='medium', color='black', weight='regular')

    st.pyplot(fig4)

    #visualisasi5
    st.subheader('Segmentasi Pelanggan Berdasarkan RFM Analysis')

    col1, col2, col3 = st.columns(3)
    with col1:
        avg_recency = round(df_rfm.recency.mean(), 1)
        st.metric("Nilai Rata-Rata Recency (hari)", value=avg_recency)
 
    with col2:
        avg_frequency = round(df_rfm.frequency.mean(), 2)
        st.metric("Nilai Rata-Rata Frequency", value=avg_frequency)
    
    with col3:
        avg_frequency = format_currency(df_rfm.monetary.mean(), "Real", locale='pt_BR') 
        st.metric("Nilai Rata-Rata Monetary", value=avg_frequency)


    fig5, ax5 = plt.subplots(nrows=1, ncols=3, figsize=(28, 8))
    sns.barplot(x="customer_unique_id", y="recency", data= df_rfm.sort_values(by='recency', ascending=True).head(10), palette='Blues_r', ax=ax5[0])
    ax5[0].bar_label(ax5[0].containers[0])
    ax5[0].set_ylabel('Hari', fontsize=12)
    ax5[0].set_xlabel('ID Unik Pelanggan')
    ax5[0].set_title("Berdasarkan Recency", loc="center", fontsize=15)
    ax5[0].tick_params(axis ='y', labelsize=15)
    ax5[0].set_xticklabels(ax5[0].get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=14)

    sns.barplot(x="customer_unique_id", y="frequency", data= df_rfm.sort_values(by='frequency', ascending=False).head(10), palette='Blues_r', ax=ax5[1])
    ax5[1].bar_label(ax5[1].containers[0])
    ax5[1].set_ylabel('Frekuensi Belanja', fontsize=12)
    ax5[1].set_xlabel('ID Unik Pelanggan')
    ax5[1].set_title("Berdasarkan Frequency", loc="center", fontsize=15)
    ax5[1].tick_params(axis='y', labelsize=15)
    ax5[1].set_xticklabels(ax5[1].get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=14)

    sns.barplot(x="customer_unique_id", y="monetary", data= df_rfm.sort_values(by='monetary', ascending=False).head(10), palette='Blues_r', ax=ax5[2])
    ax5[2].bar_label(ax5[2].containers[0])
    ax5[2].set_ylabel('Total Belanja ($)', fontsize=12)
    ax5[2].set_xlabel('ID Unik Pelanggan')
    ax5[2].set_title("Berdasarkan Monetary", loc="center", fontsize=15)
    ax5[2].tick_params(axis ='y', labelsize=15)
    ax5[2].set_xticklabels(ax5[2].get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=14)

    plt.suptitle("Pelanggan Terbaik Berdasarkan Parameter RFM", fontsize=20)
    st.pyplot(fig5)

    fig6, ax6 = plt.subplots(figsize=(6,8))
    colors = sns.color_palette(["#196ba0", "#6497b2", "#b3cde0"])
    ax6.pie(rfm_segment_df['num_cust_seg'], autopct='%1.2f%%', colors=colors, explode = [0, 0.2, 0.4])
    ax6.legend(rfm_segment_df['customer_segment'], loc='upper right', bbox_to_anchor=(1.4, 1))
    st.pyplot(fig6)

    st.caption('@latifatuzikra | 2023')

