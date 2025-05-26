import pandas as pd
import sys
import os

def load_stock_data(stock_file='data/stock_BMRI_IHSG.csv'):
    """
    Load data harga saham BMRI dan IHSG dari file CSV.

    Parameters:
    -----------
    stock_file : str
        Path ke file CSV harga saham.

    Returns:
    --------
    pd.DataFrame
        Dataframe dengan kolom tanggal dan harga saham BMRI, IHSG.
    """
    if not os.path.exists(stock_file):
        raise FileNotFoundError(f"File stock data tidak ditemukan: {stock_file}")
    try:
        data = pd.read_csv(stock_file)
        data['Date'] = pd.to_datetime(data['Date'])
        return data
    except Exception as e:
        raise RuntimeError(f"Gagal load atau parsing stock data: {e}")

def load_inflasi_data(file='data/inflation.xlsx'):
    """
    Load dan proses data inflasi dari file Excel.

    Parameters:
    -----------
    file : str
        Path ke file Excel data inflasi.

    Returns:
    --------
    pd.DataFrame
        Dataframe inflasi dengan kolom Periode (datetime) dan Inflasi (%).
    """
    bulan_map = {
        'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4,
        'Mei': 5, 'Juni': 6, 'Juli': 7, 'Agustus': 8,
        'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
    }

    def ubah_ke_datetime(bulan_tahun):
        try:
            bulan, tahun = bulan_tahun.split()
            bulan_num = bulan_map[bulan]
            return pd.Timestamp(year=int(tahun), month=bulan_num, day=1)
        except Exception:
            return pd.NaT

    if not os.path.exists(file):
        raise FileNotFoundError(f"File inflasi tidak ditemukan: {file}")
    try:
        inflasi = pd.read_excel(file)
        inflasi = inflasi.iloc[3:, 1:3]
        inflasi.columns = inflasi.iloc[0]
        inflasi = inflasi[1:]
        inflasi.reset_index(drop=True, inplace=True)
        inflasi['Periode'] = inflasi['Periode'].apply(ubah_ke_datetime)
        return inflasi
    except Exception as e:
        raise RuntimeError(f"Gagal load atau proses inflasi data: {e}")

def load_suku_bunga_data(file='data/interest_rate.xlsx'):
    """
    Load dan proses data suku bunga dari file Excel.

    Parameters:
    -----------
    file : str
        Path ke file Excel data suku bunga.

    Returns:
    --------
    pd.DataFrame
        Dataframe suku bunga dengan kolom Tanggal (datetime) dan Suku Bunga (%).
    """
    bulan_map = {
        'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4,
        'Mei': 5, 'Juni': 6, 'Juli': 7, 'Agustus': 8,
        'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
    }

    def ubah_ke_datetime_lengkap(tanggal_str):
        try:
            bagian = tanggal_str.split()
            tanggal = int(bagian[0])
            bulan = bulan_map[bagian[1]]
            tahun = int(bagian[2])
            return pd.Timestamp(year=tahun, month=bulan, day=tanggal)
        except Exception:
            return pd.NaT

    if not os.path.exists(file):
        raise FileNotFoundError(f"File suku bunga tidak ditemukan: {file}")
    try:
        suku_bunga = pd.read_excel(file)
        suku_bunga = suku_bunga.iloc[3:, 1:3]
        suku_bunga.columns = suku_bunga.iloc[0]
        suku_bunga = suku_bunga[1:]
        suku_bunga.reset_index(drop=True, inplace=True)
        suku_bunga['Tanggal'] = suku_bunga['Tanggal'].apply(ubah_ke_datetime_lengkap)
        return suku_bunga
    except Exception as e:
        raise RuntimeError(f"Gagal load atau proses suku bunga data: {e}")

def load_kurs_data(file='data/usd_to_idr.csv'):
    """
    Load dan proses data kurs USD ke IDR dari file CSV.

    Parameters:
    -----------
    file : str
        Path ke file CSV data kurs.

    Returns:
    --------
    pd.DataFrame
        Dataframe kurs dengan kolom Date (datetime) dan Price (float).
    """
    if not os.path.exists(file):
        raise FileNotFoundError(f"File kurs tidak ditemukan: {file}")
    try:
        kurs = pd.read_csv(file)
        kurs = kurs[['Date', 'Price']]
        kurs['Date'] = kurs['Date'].astype(str)
        kurs['Date'] = pd.to_datetime(kurs['Date'], format='%m/%d/%Y', errors='coerce')
        return kurs
    except Exception as e:
        raise RuntimeError(f"Gagal load atau proses kurs data: {e}")

def load_and_merge_all():
    """
    Load semua data (stock, inflasi, suku bunga, kurs) dan gabungkan ke satu dataframe.

    Returns:
    --------
    pd.DataFrame
        Dataframe gabungan dengan kolom BMRI, IHSG, Inflasi, Suku Bunga, Kurs dan Date.
    """
    try:
        data = load_stock_data()
        inflasi = load_inflasi_data()
        suku_bunga = load_suku_bunga_data()
        kurs = load_kurs_data()

        data = data.merge(inflasi, how='left', left_on='Date', right_on='Periode')
        data = data.merge(suku_bunga, how='left', left_on='Date', right_on='Tanggal')
        data = data.merge(kurs, how='left', left_on='Date', right_on='Date')

        data = data.rename(columns={
            'Close': 'BMRI',
            'Inflasi': 'Inflasi (%)',
            'Suku Bunga': 'Suku Bunga (%)',
            'USD_IDR': 'Kurs (IDR/USD)'
        })
        data = data.drop(columns=['Periode', 'Tanggal'])
        return data
    except Exception as e:
        raise RuntimeError(f"Gagal proses gabungan data: {e}")