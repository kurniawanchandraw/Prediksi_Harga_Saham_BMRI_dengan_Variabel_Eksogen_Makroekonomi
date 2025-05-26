import pandas as pd

def preprocess_data(data):
    """
    Melakukan preprocessing data gabungan harga saham BMRI dan variabel makroekonomi.
    Meliputi pengisian missing values, konversi tipe data, rename kolom,
    pengaturan index tanggal lengkap, dan interpolasi data hilang.

    Parameters:
    -----------
    data : pd.DataFrame
        Data gabungan mentah yang berisi kolom: 
        ['Date', 'BMRI', 'IHSG', 'Data Inflasi', 'BI-7Day-RR', 'Price'] (nama awal bisa bervariasi)

    Returns:
    --------
    pd.DataFrame
        Data yang sudah diproses dengan:
        - Kolom ['Date', 'BMRI', 'IHSG', 'Inflasi', 'Bunga', 'Kurs']
        - Index tanggal lengkap harian dari 2020-01-02 sampai 2025-05-01
        - Data yang sudah diinterpolasi untuk tanggal kosong
    """
    try:
        # Isi missing value pada inflasi dan suku bunga
        data['Data Inflasi'] = data['Data Inflasi'].ffill().bfill()
        data['BI-7Day-RR'] = data['BI-7Day-RR'].ffill().bfill()

        # Konversi persentase string ke float desimal
        data['Data Inflasi'] = data['Data Inflasi'].str.replace('%', '').str.strip().astype(float) / 100
        data['BI-7Day-RR'] = data['BI-7Day-RR'].str.replace('%', '').str.strip().astype(float) / 100

        # Ubah kolom Kurs (Price) dari string ke float
        data['Price'] = data['Price'].str.replace(',', '').astype(float)

        # Rename kolom
        columns = ['Date', 'BMRI', 'IHSG', 'Inflasi', 'Bunga', 'Kurs']
        data.columns = columns

        # Konversi kolom Date ke datetime dan set sebagai index
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)

        # Buat index tanggal lengkap harian
        tanggal_lengkap = pd.date_range(start='2020-01-02', end='2025-05-01', freq='D')
        data = data.reindex(tanggal_lengkap)

        # Interpolasi data kosong berdasarkan waktu
        data_interpolated = data.interpolate(method='time')

        # Reset index ke kolom Date
        data_interpolated = data_interpolated.reset_index().rename(columns={'index': 'Date'})

        return data_interpolated

    except KeyError as ke:
        raise KeyError(f"Kolom yang diperlukan tidak ditemukan di data: {ke}")
    except AttributeError as ae:
        raise AttributeError(f"Tipe data tidak sesuai untuk operasi string: {ae}")
    except Exception as e:
        raise RuntimeError(f"Terjadi kesalahan saat preprocessing data: {e}")