import pandas as pd
import numpy as np
import re
import logging
import math
import os
import pickle
#from config import Config
import time
from datetime import datetime, timedelta
from typing import Union, Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# Logger ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Config:
    """
    Uygulama için tüm konfigürasyonları ve sabitleri barındırır.
    Puanlama ağırlıklarını ve dosya yollarını buradan kolayca değiştirebilirsiniz.
    """
    # --- Dosya Yolları ---
    # Lütfen bu yolları kendi sisteminize göre güncelleyin.
    DATASET_PATHS = [
        'data/vatan_laptop_data_cleaned.csv',
        'data/amazon_final.csv', 
        'data/cleaned_incehesap_data.csv'
    ]

    # Cache ayarları
    CACHE_FILE = 'data/laptop_cache.pkl'
    CACHE_DURATION = timedelta(hours=24)

    # --- Veri Temizleme Sabitleri ---
    # Genişletilmiş GPU skorları
    GPU_SCORES = {
        'rtx5090': 110, 'rtx5080': 105, 'rtx5070': 100, 'rtx5060': 85, 'rtx5050': 75,
        'rtx5000': 96, 'rtx4090': 100, 'rtx4080': 95, 'rtx3080': 88, 'rtx4070': 85,
        'rtx3070': 80, 'rtx4060': 75, 'rtx3060': 70, 'rtx4050': 60,
        'rtx3050': 55, 'rtx2060': 50, 'rtx': 45, 'gtx': 40,
        'mx550': 38, 'intel arc': 45, 'apple integrated': 35, 'intel uhd': 22,
        'intel iris xe graphics': 25, 'iris xe': 25, 'integrated': 20,
        'unknown': 30
    }
    DEFAULT_GPU_SCORE = 30  # Tanımlanamayan harici kartlar için varsayılan puan

    # Genişletilmiş CPU skorları
    CPU_SCORES = {
        'ultra 9 275hx': 98, 'ultra 9': 100, 'ultra 7 255h': 92, 'ultra 7': 90,
        'ultra 5 155h': 83, 'ultra 5': 80, 'core ultra 9': 98, 'core ultra 7': 90,
        'core ultra 5': 83, 'ryzen ai 9 hx370': 95, 'core 5 210h': 75,
        'i9': 95, 'i7': 85, 'i5': 75, 'i3': 60,
        'ryzen 9': 95, 'ryzen 7': 85, 'ryzen 5': 75, 'ryzen 3': 60,
        'm4 pro': 94, 'm4': 88, 'm3': 85, 'm2': 80, 'm1': 75,
        'snapdragon x': 78, 'unknown': 50
    }

    # Marka güvenilirlik skorları
    BRAND_SCORES = {
        'apple': 0.95, 'dell': 0.85, 'hp': 0.80, 'lenovo': 0.85,
        'asus': 0.82, 'msi': 0.80, 'acer': 0.75, 'monster': 0.70,
        'huawei': 0.78, 'samsung': 0.83, 'lg': 0.77, 'gigabyte': 0.76
    }

    # --- Puanlama Ağırlıkları ---
    WEIGHTS = {
        'price_fit': 15,
        'price_performance': 10,
        'purpose': {
            'base': 30,
            'oyun': {'dedicated': 1.0, 'integrated': 0.1, 'apple': 0.5},
            'taşınabilirlik': {'dedicated': 0.2, 'integrated': 1.0, 'apple': 0.9},
            'üretkenlik': {'dedicated': 0.6, 'integrated': 0.4, 'apple': 1.0},
            'tasarım': {'dedicated': 0.8, 'integrated': 0.5, 'apple': 1.0},
        },
        'user_preferences': {
            'performance': 12,
            'battery': 12,
            'portability': 8,
        },
        'specs': {
            'ram': 5,
            'ssd': 5,
        },
        'brand_reliability': 8,
    }

    # Ekran boyutu kategorileri
    SCREEN_CATEGORIES = {
        1: (13, 14),  # Kompakt
        2: (15, 16),  # Standart
        3: (17, 18),  # Büyük
        4: (0, 99)  # Farketmez
    }


class CachedDataHandler:
    """Önbellekli veri yükleme ve birleştirme işlemlerini yönetir."""

    def __init__(self, config: Config):
        self.config = config

    def _is_cache_valid(self) -> bool:
        """Cache dosyasının geçerliliğini kontrol et"""
        if not os.path.exists(self.config.CACHE_FILE):
            return False

        cache_time = datetime.fromtimestamp(os.path.getmtime(self.config.CACHE_FILE))
        return datetime.now() - cache_time < self.config.CACHE_DURATION

    def _load_from_cache(self) -> pd.DataFrame:
        """Cache'den veri yükle"""
        with open(self.config.CACHE_FILE, 'rb') as f:
            return pickle.load(f)

    def _save_to_cache(self, df: pd.DataFrame):
        """Veriyi cache'e kaydet"""
        with open(self.config.CACHE_FILE, 'wb') as f:
            pickle.dump(df, f)

    def load_and_merge_data(self) -> pd.DataFrame:
        """CSV dosyalarını yükler, birleştirir ve tekilleştirir."""
        if self._is_cache_valid():
            logger.info("Önbellekten veri yükleniyor...")
            return self._load_from_cache()

        datasets = []
        for i, path in enumerate(self.config.DATASET_PATHS, 1):
            try:
                df = pd.read_csv(
                    path,
                    quotechar='"',
                    doublequote=True,
                    escapechar='\\',
                    engine='python'
                )
                df['data_source'] = f'dataset_{i}'
                datasets.append(df)
                logger.info(f"Dataset {i} yüklendi: {len(df)} satır")
            except FileNotFoundError:
                logger.warning(f"'{path}' dosyası bulunamadı. Bu kaynak atlanıyor.")
            except Exception as e:
                logger.error(f"'{path}' dosyası yüklenirken hata: {e}")

        if not datasets:
            raise FileNotFoundError("Hiçbir veri dosyası bulunamadı.")

        combined_df = pd.concat(datasets, ignore_index=True)

        # Duplike verileri temizle
        initial_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['name', 'price'], keep='first')
        logger.info(f"Veriler birleştirildi. {initial_count - len(combined_df)} duplike satır kaldırıldı.")

        # Cache'e kaydet
        self._save_to_cache(combined_df)

        return combined_df


class EnhancedDataHandler:
    """Gelişmiş veri temizleme ve işleme fonksiyonları."""

    def __init__(self, config: Config):
        self.config = config

    @staticmethod
    def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = (
            df.columns.str.strip().str.lower()
            .str.replace(r'[\s\-]+', '_', regex=True)
        )
        return df

    @staticmethod
    def _clean_price(series: pd.Series) -> pd.Series:
        """Fiyat temizleme"""

        def clean_single_price(val):
            if pd.isna(val):
                return None

            price_str = str(val).strip()
            if not price_str or price_str.upper() in ['NAN', 'NULL', '']:
                return None

            # Para birimi işaretlerini temizle
            price_str = re.sub(r'(TL|₺|TRY|LIRA|LİRA)', '', price_str, flags=re.IGNORECASE)
            price_str = re.sub(r'[^\d,.]', '', price_str)

            # Binlik ayraç ve ondalık düzeltme
            if '.' in price_str and ',' in price_str:
                if price_str.rfind('.') > price_str.rfind(','):
                    price_str = price_str.replace(',', '').replace('.', ',')
                else:
                    price_str = price_str.replace('.', '')

            price_str = price_str.replace(',', '.')

            try:
                price_float = float(price_str)
                return int(price_float)
            except ValueError:
                return None

        return series.apply(clean_single_price)

    @staticmethod
    def _clean_screen_size(series: pd.Series) -> pd.Series:
        """Ekran boyutu temizleme"""

        def clean_single_screen(val):
            if pd.isna(val):
                return None

            screen_str = str(val).strip()
            screen_str = screen_str.replace('"', '').replace("'", '')

            match = re.search(r'(\d+(?:\.\d+)?)', screen_str)
            if match:
                return float(match.group(1))
            return None

        return series.apply(clean_single_screen)

    @staticmethod
    def _normalize_storage_ram(val: Union[str, float, int], is_ssd: bool = True) -> Optional[int]:
        """SSD ve RAM normalizasyonu"""
        if pd.isna(val):
            return None

        val_str = str(val).upper().strip()
        if not val_str or val_str in ['NAN', 'NULL', '']:
            return None

        # TB dönüşümü
        tb_match = re.search(r'(\d+(?:\.\d+)?)\s*TB', val_str)
        if tb_match:
            tb_value = float(tb_match.group(1))
            return int(tb_value * 1024)

        # GB dönüşümü
        gb_match = re.search(r'(\d+(?:\.\d+)?)\s*GB', val_str)
        if gb_match:
            gb_value = float(gb_match.group(1))
            return int(gb_value)

        # MB dönüşümü
        mb_match = re.search(r'(\d+(?:\.\d+)?)\s*MB', val_str)
        if mb_match:
            mb_value = float(mb_match.group(1))
            return math.ceil(mb_value / 1024)

        # Sadece sayı varsa
        number_match = re.search(r'(\d+)', val_str)
        if number_match:
            return int(number_match.group(1))

        return None

    @staticmethod
    def _normalize_url(url: str) -> str:
        """URL normalizasyonu"""
        if pd.isna(url) or not str(url).strip():
            return url

        url_str = str(url).strip()
        if not url_str.startswith(('http://', 'https://')):
            return f'https://{url_str}'
        return url_str

    def _normalize_gpu(self, gpu_str: str) -> str:
        """GPU normalizasyonu"""
        if pd.isna(gpu_str) or not str(gpu_str).strip():
            return 'unknown'

        g_low = str(gpu_str).lower()

        for key in sorted(self.config.GPU_SCORES.keys(), key=len, reverse=True):
            if key in g_low:
                if 'rtx' in key and re.search(r'rtx\s*(\d{4})', g_low):
                    rtx_match = re.search(r'rtx\s*(\d{4})', g_low)
                    rtx_model = f"rtx{rtx_match.group(1)}"
                    if rtx_model in self.config.GPU_SCORES:
                        return rtx_model
                return key

        return 'unknown'

    def _normalize_cpu(self, cpu_str: str) -> str:
        """CPU normalizasyonu"""
        if pd.isna(cpu_str) or not str(cpu_str).strip():
            return 'unknown'

        c_low = str(cpu_str).lower()

        for key in sorted(self.config.CPU_SCORES.keys(), key=len, reverse=True):
            if key in c_low:
                return key

        return 'unknown'

    def _detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aykırı değerleri tespit et"""
        # Fiyat aykırıları
        Q1 = df['price'].quantile(0.25)
        Q3 = df['price'].quantile(0.75)
        IQR = Q3 - Q1

        price_outliers = (df['price'] < Q1 - 1.5 * IQR) | (df['price'] > Q3 + 1.5 * IQR)
        df['is_price_outlier'] = price_outliers

        # Mantıksız kombinasyonları tespit et
        suspicious = (
                (df['price'] < 10000) &
                (df['gpu_score'] > 80) &
                (df['ram_gb'] >= 16)
        )
        df['is_suspicious'] = suspicious

        return df

    def _normalize_model_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Model isimlerini standartlaştır"""
        replacements = {
            r'\bNB\b': 'Notebook',
            r'\bLT\b': 'Laptop',
            r'\bGB\b': 'GB',
            r'\bTB\b': 'TB',
        }

        for pattern, replacement in replacements.items():
            df['name'] = df['name'].str.replace(pattern, replacement, regex=True)

        return df

    def _extract_brand(self, name: str) -> str:
        """İsimden marka çıkar"""
        name_lower = str(name).lower()
        for brand in self.config.BRAND_SCORES.keys():
            if brand in name_lower:
                return brand
        return 'other'

    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """DataFrame'i temizle ve standardize et."""
        df = self._clean_column_names(df)
        df = self._normalize_model_names(df)

        stats = {
            'rows_in': len(df),
            'rows_out': 0,
            'invalid_url': 0,
            'outliers_detected': 0,
            'suspicious_items': 0
        }

        # Gerekli sütunları kontrol et
        required_cols = ['url', 'name', 'price', 'screen_size', 'ssd', 'cpu', 'ram', 'os', 'gpu']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"'{col}' sütunu veri setinde bulunamadı.")

        # Temizleme işlemleri
        df['price'] = self._clean_price(df['price'])
        df['screen_size'] = self._clean_screen_size(df['screen_size'])
        df['ssd_gb'] = df['ssd'].apply(lambda x: self._normalize_storage_ram(x, is_ssd=True))
        df['ram_gb'] = df['ram'].apply(lambda x: self._normalize_storage_ram(x, is_ssd=False))

        # URL düzeltme
        original_urls = df['url'].copy()
        df['url'] = df['url'].apply(self._normalize_url)
        stats['invalid_url'] = sum(~original_urls.str.startswith(('http://', 'https://'), na=True))

        # İşletim sistemi normalizasyonu
        df['os'] = df['os'].astype(str).str.strip().str.upper().replace({'NAN': np.nan, '': np.nan})

        # GPU ve CPU normalizasyonu
        df['gpu_clean'] = df['gpu'].apply(self._normalize_gpu)
        df['cpu_clean'] = df['cpu'].apply(self._normalize_cpu)

        # Marka çıkarımı
        df['brand'] = df['name'].apply(self._extract_brand)

        # Puanlama için yardımcı sütunlar
        df['is_apple'] = df['brand'] == 'apple'
        df['has_dedicated_gpu'] = ~df['gpu_clean'].isin(
            ['integrated', 'apple integrated', 'iris xe', 'intel uhd', 'unknown'])
        df['gpu_score'] = df['gpu_clean'].apply(lambda x: self.config.GPU_SCORES.get(x, self.config.DEFAULT_GPU_SCORE))
        df['cpu_score'] = df['cpu_clean'].apply(lambda x: self.config.CPU_SCORES.get(x, 50))
        df['brand_score'] = df['brand'].apply(lambda x: self.config.BRAND_SCORES.get(x, 0.70))

        # Aykırı değer tespiti
        df = self._detect_outliers(df)
        stats['outliers_detected'] = df['is_price_outlier'].sum()
        stats['suspicious_items'] = df['is_suspicious'].sum()

        # Kullanılacak sütunları seç
        final_cols = [
            'url', 'name', 'brand', 'price', 'screen_size', 'ssd_gb', 'ram_gb', 'os',
            'gpu_clean', 'cpu_clean', 'gpu_score', 'cpu_score', 'brand_score',
            'is_apple', 'has_dedicated_gpu', 'is_price_outlier', 'is_suspicious', 'data_source'
        ]
        df = df[final_cols]

        # Kritik eksik verileri kaldır
        df = df.dropna(subset=['price', 'ram_gb', 'ssd_gb'])
        stats['rows_out'] = len(df)

        logger.info(f"Veri temizleme tamamlandı. İstatistikler: {stats}")

        return df, stats


class AdvancedScoringEngine:
    """Gelişmiş puanlama algoritmaları - DÜZELTME."""

    def __init__(self, config):
        self.config = config
        self.weights = self.config.WEIGHTS

    def _calculate_price_performance_ratio(self, row: pd.Series, preferences: Dict[str, Any]) -> float:
        """Fiyat/performans oranını hesapla"""
        try:
            # GPU ve CPU skorlarını birleştir
            performance_score = (row['gpu_score'] * 0.6 + row['cpu_score'] * 0.4) / 100

            # Fiyat/performans oranı
            price_ratio = preferences['ideal_price'] / row['price'] if row['price'] > 0 else 0
            return performance_score * price_ratio * self.weights['price_performance']
        except Exception as e:
            logger.warning(f"Fiyat/performans hesaplama hatası: {e}")
            return 0.0

    def _calculate_purpose_score(self, row: pd.Series, purpose: str) -> float:
        """Kullanım amacına göre temel puanı hesaplar."""
        try:
            purpose_weights = self.weights['purpose'][purpose]

            if row['is_apple']:
                multiplier = purpose_weights['apple']
            elif row['has_dedicated_gpu']:
                multiplier = purpose_weights['dedicated']
            else:
                multiplier = purpose_weights['integrated']

            # GPU ve CPU performansının birleşik etkisi
            combined_performance = (row['gpu_score'] * 0.7 + row['cpu_score'] * 0.3) / 100
            return self.weights['purpose']['base'] * combined_performance * multiplier
        except Exception as e:
            logger.warning(f"Amaç puanı hesaplama hatası: {e}")
            return 0.0

    def _calculate_user_preference_score(self, row: pd.Series, prefs: Dict[str, Any]) -> float:
        """Kullanıcı tercihlerine göre puan hesapla."""
        try:
            score = 0
            w = self.weights['user_preferences']

            # Performans
            perf_score = (row['gpu_score'] * 0.6 + row['cpu_score'] * 0.4) / 100
            score += w['performance'] * perf_score * (prefs['performance_importance'] / 5)

            # Pil & Taşınabilirlik
            portability_factor = 1.0
            if row['is_apple']:
                portability_factor = 0.9
            elif row['has_dedicated_gpu']:
                portability_factor = 0.4
            elif row['screen_size'] <= 14:
                portability_factor = 1.2

            score += w['battery'] * portability_factor * (prefs['battery_importance'] / 5)
            score += w['portability'] * portability_factor * (prefs['portability_importance'] / 5)

            return score
        except Exception as e:
            logger.warning(f"Kullanıcı tercihi puanı hesaplama hatası: {e}")
            return 0.0

    def _apply_filters(self, df: pd.DataFrame, preferences: Dict[str, Any]) -> pd.DataFrame:
        """Kullanıcı filtrelerini uygula"""
        try:
            filtered_df = df.copy()

            # Fiyat filtresi
            filtered_df = filtered_df[
                (filtered_df['price'] >= preferences['min_budget']) &
                (filtered_df['price'] <= preferences['max_budget'])
            ]

            # Ekran boyutu filtresi
            if preferences.get('screen_preference') and preferences['screen_preference'] != 4:
                min_size, max_size = self.config.SCREEN_CATEGORIES[preferences['screen_preference']]
                filtered_df = filtered_df[
                    (filtered_df['screen_size'] >= min_size) &
                    (filtered_df['screen_size'] <= max_size)
                ]

            # İşletim sistemi filtresi
            if preferences.get('os_preference'):
                if preferences['os_preference'] == 1:  # Windows
                    filtered_df = filtered_df[filtered_df['os'].str.contains('WINDOWS', na=False)]
                elif preferences['os_preference'] == 2:  # macOS
                    filtered_df = filtered_df[filtered_df['is_apple'] == True]

            # Marka filtresi
            if preferences.get('brand_preference'):
                filtered_df = filtered_df[filtered_df['brand'] == preferences['brand_preference']]

            # Minimum donanım gereksinimleri
            if preferences.get('min_ram'):
                filtered_df = filtered_df[filtered_df['ram_gb'] >= preferences['min_ram']]
            if preferences.get('min_ssd'):
                filtered_df = filtered_df[filtered_df['ssd_gb'] >= preferences['min_ssd']]

            # Şüpheli ürünleri çıkar
            filtered_df = filtered_df[filtered_df['is_suspicious'] == False]

            return filtered_df
        except Exception as e:
            logger.error(f"Filtreleme hatası: {e}")
            return pd.DataFrame()

    def calculate_scores_parallel(self, df: pd.DataFrame, preferences: Dict[str, Any]) -> pd.DataFrame:
        """Paralel puanlama işlemi - DÜZELTME."""
        # Filtreleri uygula
        filtered_df = self._apply_filters(df, preferences)

        if filtered_df.empty:
            print("❌ Filtreleme sonrası hiç laptop bulunamadı!")
            return pd.DataFrame()

        def calculate_row_score(row: pd.Series) -> float:
            try:
                # 1. Fiyat Uygunluğu Puanı
                price_range = preferences['max_budget'] - preferences['min_budget']
                if price_range > 0:
                    price_diff = abs(row['price'] - preferences['ideal_price'])
                    price_score = self.weights['price_fit'] * max(0, 1 - price_diff / (price_range / 2))
                else:
                    price_score = self.weights['price_fit']

                # 2. Fiyat/Performans Puanı
                price_perf_score = self._calculate_price_performance_ratio(row, preferences)

                # 3. Kullanım Amacı Puanı
                purpose_score = self._calculate_purpose_score(row, preferences['purpose'])

                # 4. Kullanıcı Tercihleri Puanı
                user_pref_score = self._calculate_user_preference_score(row, preferences)

                # 5. Donanım Puanları
                ram_score = self.weights['specs']['ram'] * min(row['ram_gb'] / 16, 1.0)
                ssd_score = self.weights['specs']['ssd'] * min(row['ssd_gb'] / 1024, 1.0)

                # 6. Marka Güvenilirlik Puanı
                brand_score = self.weights['brand_reliability'] * row['brand_score']

                # Toplam puan
                total_score = (price_score + price_perf_score + purpose_score +
                               user_pref_score + ram_score + ssd_score + brand_score)

                # ÖNEMLİ: Kesinlikle float döndür!
                result = float(max(0, min(100, total_score)))
                return result
                
            except Exception as e:
                logger.warning(f"Puan hesaplama hatası: {e}")
                return 0.0

        # Score hesaplama ve atama - DÜZELTME
        print(f"Filtreleme sonrası {len(filtered_df)} laptop bulundu...")
        
        # Skorları hesapla
        if len(filtered_df) < 100:
            # Tek tek hesapla
            scores = []
            for idx, row in filtered_df.iterrows():
                score = calculate_row_score(row)
                scores.append(score)
                
            # Score sütununu ekle - NUMERIC olduğundan emin ol
            filtered_df = filtered_df.copy()
            filtered_df['score'] = scores
            
            # Veri tipini kontrol et ve düzelt
            filtered_df['score'] = pd.to_numeric(filtered_df['score'], errors='coerce')
            filtered_df['score'] = filtered_df['score'].fillna(0.0)
            
        else:
            # Paralel işleme
            chunks = np.array_split(filtered_df, multiprocessing.cpu_count())
            
            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = []
                for chunk in chunks:
                    future = executor.submit(lambda c: [calculate_row_score(row) for _, row in c.iterrows()], chunk)
                    futures.append(future)

                all_scores = []
                for future in futures:
                    chunk_scores = future.result()
                    all_scores.extend(chunk_scores)

            filtered_df = filtered_df.copy()
            filtered_df['score'] = all_scores
            filtered_df['score'] = pd.to_numeric(filtered_df['score'], errors='coerce').fillna(0.0)

        logger.info(f"Filtreleme sonrası {len(filtered_df)} laptop için puanlama yapıldı.")
        return filtered_df


class MLRecommender:
    """Makine öğrenmesi tabanlı öneri sistemi."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = NearestNeighbors(n_neighbors=10, metric='cosine')

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Özellikleri ML için hazırla"""
        # Sayısal özellikler
        numeric_features = df[[
            'price', 'screen_size', 'ssd_gb', 'ram_gb',
            'gpu_score', 'cpu_score', 'brand_score'
        ]].fillna(0)

        # Kategorik özellikler için one-hot encoding
        categorical_features = pd.get_dummies(df[['brand', 'gpu_clean', 'cpu_clean']],
                                              prefix=['brand', 'gpu', 'cpu'])

        # Özellikleri birleştir
        features = pd.concat([numeric_features, categorical_features], axis=1)

        return self.scaler.fit_transform(features)

    def find_similar_laptops(self, target_idx: int, df: pd.DataFrame, n_recommendations: int = 5):
        """Benzer laptopları bul"""
        features = self.prepare_features(df)
        self.model.fit(features)

        # Hedef laptop'un özelliklerini al
        target_features = features[target_idx].reshape(1, -1)

        # En yakın komşuları bul
        distances, indices = self.model.kneighbors(target_features, n_neighbors=n_recommendations + 1)

        # İlk sonuç kendisi olacağı için onu çıkar
        similar_indices = indices[0][1:]
        similar_distances = distances[0][1:]

        similar_df = df.iloc[similar_indices].copy()
        similar_df['similarity_score'] = 1 - similar_distances  # Benzerlik skoru

        return similar_df


class TrendAnalyzer:
    """Fiyat ve pazar trend analizi - %20 fırsat sistemi ile."""

    def __init__(self, min_price_threshold=1000, discount_threshold=20.0):
        # Anomali tespiti için model
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(
            contamination=0.1,  # En iyi %10'luk fırsatları bul
            random_state=42,
            n_estimators=100
        )
        # Minimum fiyat eşiği ve indirim eşiği
        self.min_price_threshold = min_price_threshold
        self.discount_threshold = discount_threshold  # %20 varsayılan

    def find_deal_products(self, df: pd.DataFrame) -> pd.DataFrame:
        """%20+ indirimli ürünleri fırsat olarak tespit et"""
        logger.info(f"Fırsat ürünleri aranıyor (min %{self.discount_threshold} indirim)...")
        
        # Önce temel filtrelemeyi yap - minimum fiyat ve geçerli veriler
        clean_df = df.dropna(subset=['price', 'gpu_score', 'cpu_score', 'ram_gb', 'ssd_gb']).copy()
        
        # Minimum fiyat filtresi
        clean_df = clean_df[clean_df['price'] >= self.min_price_threshold]
        
        # Ayrıca açıkça hatalı görünen fiyatları da çıkar
        clean_df = clean_df[~(
            (clean_df['gpu_score'] >= 90) & (clean_df['price'] < 15000)  # Üst seviye GPU'lar çok ucuz olamaz
        )]
        
        # Mantıksız RAM/SSD kombinasyonlarını filtrele
        clean_df = clean_df[~(
            (clean_df['ram_gb'] >= 32) & (clean_df['price'] < 8000)  # 32GB+ RAM çok ucuz olamaz
        )]
        
        if len(clean_df) < 10:
            logger.warning("Yeterli veri bulunamadı")
            return pd.DataFrame()

        # Özellik mühendisliği
        clean_df['performance_score'] = (clean_df['gpu_score'] * 0.6 + clean_df['cpu_score'] * 0.4)
        clean_df['spec_score'] = (
                (clean_df['ram_gb'] / 32) * 30 +  # RAM skoru (max 32GB varsayımı)
                (clean_df['ssd_gb'] / 2048) * 30 +  # SSD skoru (max 2TB varsayımı)
                clean_df['performance_score'] * 0.4
        )

        # Her ürün için piyasa fiyatını hesapla
        logger.info("Piyasa fiyatları hesaplanıyor...")
        clean_df['market_price'] = clean_df.apply(
            lambda row: self._calculate_market_price(row, clean_df), axis=1
        )
        
        # İndirim oranını hesapla
        clean_df['discount_percentage'] = (
            (clean_df['market_price'] - clean_df['price']) / clean_df['market_price'] * 100
        ).round(1)
        
        # Negatif indirimleri 0 yap (piyasadan pahalı olanlar)
        clean_df['discount_percentage'] = clean_df['discount_percentage'].clip(lower=0)
        
        # %20+ indirimli ürünleri fırsat olarak işaretle
        deals = clean_df[
            clean_df['discount_percentage'] >= self.discount_threshold
        ].copy()

        if deals.empty:
            logger.info(f"Min %{self.discount_threshold} indirimli ürün bulunamadı")
            return pd.DataFrame()
        
        # Fırsat skoru hesapla (indirim oranına dayalı)
        deals['deal_score'] = self._calculate_deal_score(deals)
        
        # Ek güvenlik kontrolleri
        deals = self._apply_safety_checks(deals)
        
        # Son validasyon
        deals = deals[deals['discount_percentage'] >= self.discount_threshold]
        
        if not deals.empty:
            logger.info(f"✅ {len(deals)} fırsat ürün tespit edildi (min %{self.discount_threshold} indirim)")
            # Sırala ve döndür
            return deals.sort_values('deal_score', ascending=False)
        else:
            logger.info("Final kontrol sonrası hiç fırsat ürün kalmadı")
            return pd.DataFrame()

    def _calculate_market_price(self, target_row: pd.Series, df: pd.DataFrame) -> float:
        """Belirli bir ürün için piyasa fiyatını hesapla"""
        
        # 1. AŞAMA: Çok benzer ürünleri bul (dar kriter)
        similar_narrow = df[
            (df['performance_score'].between(
                target_row['performance_score'] * 0.85,
                target_row['performance_score'] * 1.15
            )) &
            (df['ram_gb'].between(
                max(4, target_row['ram_gb'] - 4),
                target_row['ram_gb'] + 4
            )) &
            (df['ssd_gb'].between(
                max(128, target_row['ssd_gb'] - 256),
                target_row['ssd_gb'] + 256
            )) &
            (df['name'] != target_row['name']) &  # Kendisini hariç tut
            (df['price'] >= self.min_price_threshold)
        ]
        
        if len(similar_narrow) >= 3:
            # Aykırı değerleri temizle
            market_price = self._clean_outliers_and_average(similar_narrow['price'])
            if market_price > 0:
                return market_price
        
        # 2. AŞAMA: Orta benzerlik (geniş kriter)
        similar_broad = df[
            (df['performance_score'].between(
                target_row['performance_score'] * 0.7,
                target_row['performance_score'] * 1.3
            )) &
            (df['ram_gb'].between(
                max(4, target_row['ram_gb'] - 8),
                target_row['ram_gb'] + 8
            )) &
            (df['price'] >= self.min_price_threshold) &
            (df['name'] != target_row['name'])
        ]
        
        if len(similar_broad) >= 5:
            market_price = self._clean_outliers_and_average(similar_broad['price'])
            if market_price > 0:
                return market_price
        
        # 3. AŞAMA: Aynı kategori (çok geniş)
        category_similar = df[
            (df['gpu_score'].between(
                target_row['gpu_score'] - 20,
                target_row['gpu_score'] + 20
            )) &
            (df['cpu_score'].between(
                target_row['cpu_score'] - 20,
                target_row['cpu_score'] + 20
            )) &
            (df['price'] >= self.min_price_threshold) &
            (df['name'] != target_row['name'])
        ]
        
        if len(category_similar) >= 3:
            market_price = self._clean_outliers_and_average(category_similar['price'])
            if market_price > 0:
                return market_price
        
        # 4. AŞAMA: Son çare - ürünün kendi fiyatı
        logger.warning(f"Piyasa fiyatı bulunamadı: {target_row['name'][:50]}...")
        return target_row['price']

    def _clean_outliers_and_average(self, prices: pd.Series) -> float:
        """Fiyat listesindeki aykırı değerleri temizle ve ortalama al"""
        if len(prices) < 2:
            return prices.mean() if len(prices) > 0 else 0
        
        # IQR yöntemi ile aykırı değerleri temizle
        Q1 = prices.quantile(0.25)
        Q3 = prices.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR > 0:
            # Aykırı değerleri filtrele
            filtered_prices = prices[
                (prices >= Q1 - 1.5 * IQR) &
                (prices <= Q3 + 1.5 * IQR)
            ]
            
            if len(filtered_prices) >= 2:
                return filtered_prices.mean()
        
        # IQR uygulanamıyorsa median kullan
        return prices.median()

    def _calculate_deal_score(self, deals_df: pd.DataFrame) -> pd.Series:
        """Fırsat skorunu hesapla (0-100 arası)"""
        if deals_df.empty:
            return pd.Series(dtype=float)
        
        # İndirim oranı bazlı puan (ana faktör)
        discount_score = deals_df['discount_percentage'].clip(0, 70)  # Max %70
        
        # Performans bonusu (yüksek performanslı ürünler daha değerli)
        performance_bonus = (deals_df['performance_score'] / 100) * 10
        
        # Spec bonusu (RAM + SSD)
        spec_bonus = (
            (deals_df['ram_gb'] / 32) * 5 +  # RAM bonusu
            (deals_df['ssd_gb'] / 1024) * 3   # SSD bonusu
        ).clip(0, 8)
        
        # Toplam skor
        total_score = discount_score + performance_bonus + spec_bonus
        
        # 0-100 arası normalize et
        return total_score.clip(0, 100).round(1)

    def _apply_safety_checks(self, deals_df: pd.DataFrame) -> pd.DataFrame:
        """Güvenlik kontrolleri uygula"""
        if deals_df.empty:
            return deals_df
        
        # 1. Çok aşırı indirimleri şüpheli say (%70'ten fazla)
        deals_df = deals_df[deals_df['discount_percentage'] <= 70]
        
        # 2. Çok düşük fiyatlı ürünleri kontrol et
        deals_df = deals_df[~(
            (deals_df['performance_score'] >= 80) & 
            (deals_df['price'] < 8000)
        )]
        
        # 3. Mantıksız fiyat/özellik oranlarını filtrele
        deals_df = deals_df[~(
            (deals_df['ram_gb'] >= 32) & 
            (deals_df['ssd_gb'] >= 1024) & 
            (deals_df['price'] < 15000)
        )]
        
        return deals_df

    def get_deal_insights(self, deals_df: pd.DataFrame) -> Dict[str, Any]:
        """Fırsat ürünleri hakkında içgörüler"""
        if deals_df.empty:
            return {
                'total_deals': 0,
                'message': f'Min %{self.discount_threshold} indirimli ürün bulunamadı'
            }

        insights = {
            'total_deals': len(deals_df),
            'avg_discount': deals_df['discount_percentage'].mean(),
            'max_discount': deals_df['discount_percentage'].max(),
            'min_discount': deals_df['discount_percentage'].min(),
            'best_deal_category': deals_df.loc[deals_df['deal_score'].idxmax(), 'gpu_clean'] if not deals_df.empty else 'N/A',
            'price_range': (deals_df['price'].min(), deals_df['price'].max()),
            'top_brands': deals_df['brand'].value_counts().head(3).to_dict() if 'brand' in deals_df.columns else {},
            'discount_threshold': f"%{self.discount_threshold}",
            'min_price_filter': f"{self.min_price_threshold:,} TL",
            'avg_deal_score': deals_df['deal_score'].mean(),
            'savings_info': {
                'total_savings': (deals_df['market_price'] - deals_df['price']).sum(),
                'avg_savings': (deals_df['market_price'] - deals_df['price']).mean(),
                'max_savings': (deals_df['market_price'] - deals_df['price']).max()
            }
        }

        return insights

    def analyze_price_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fiyat trendlerini analiz et"""
        # Minimum fiyat filtresi uygula
        filtered_df = df[df['price'] >= self.min_price_threshold]
        
        trends = {}

        # GPU bazlı ortalama fiyatlar
        gpu_prices = filtered_df.groupby('gpu_clean')['price'].agg(['mean', 'min', 'max', 'count'])
        gpu_prices = gpu_prices[gpu_prices['count'] >= 5]  # En az 5 ürün olan GPU'lar
        trends['gpu_prices'] = gpu_prices.round(0).to_dict('index')

        # Marka bazlı ortalama fiyatlar
        brand_prices = filtered_df.groupby('brand')['price'].agg(['mean', 'min', 'max', 'count'])
        brand_prices = brand_prices[brand_prices['count'] >= 3]
        trends['brand_prices'] = brand_prices.round(0).to_dict('index')

        # RAM/SSD kombinasyonlarının fiyat etkileri
        filtered_df['config'] = filtered_df['ram_gb'].astype(str) + 'GB RAM / ' + filtered_df['ssd_gb'].astype(str) + 'GB SSD'
        config_prices = filtered_df.groupby('config')['price'].mean().sort_values()
        trends['config_prices'] = config_prices.round(0).to_dict()

        # Fiyat aralıklarına göre dağılım
        price_ranges = pd.cut(filtered_df['price'], bins=[self.min_price_threshold, 20000, 30000, 40000, 50000, np.inf],
                              labels=[f'{self.min_price_threshold//1000}K-20K', '20-30K', '30-40K', '40-50K', '50K+'])
        trends['price_distribution'] = filtered_df.groupby(price_ranges).size().to_dict()

        return trends

    def suggest_best_time_to_buy(self, category: str) -> str:
        """Alım için en iyi zamanı öner"""
        seasonal_discounts = {
            'oyun': "Black Friday (Kasım), yılbaşı ve okul dönemi başlangıcında büyük indirimler oluyor.",
            'taşınabilirlik': "Ağustos-Eylül okul sezonu ve Mart-Nisan bahar kampanyaları idealdir.",
            'üretkenlik': "Yıl sonu kurumsal bütçe dönemlerinde ve Mayıs ayında fırsatlar artıyor.",
            'tasarım': "Adobe Creative Cloud kampanyaları ile eş zamanlı dönemlerde indirimler oluyor."
        }
        return seasonal_discounts.get(category, "Genel olarak Black Friday ve yılbaşı dönemlerinde.")

    def get_market_insights(self, df: pd.DataFrame, preferences: Dict[str, Any]) -> Dict[str, str]:
        """Pazar içgörüleri oluştur"""
        # Minimum fiyat filtresi uygula
        filtered_df = df[df['price'] >= self.min_price_threshold]
        
        insights = {}

        # Bütçe analizi
        budget_laptops = filtered_df[(filtered_df['price'] >= preferences['min_budget']) &
                            (filtered_df['price'] <= preferences['max_budget'])]

        avg_price = budget_laptops['price'].mean()
        insights[
            'budget_analysis'] = f"Bütçenizde {len(budget_laptops)} laptop var. Ortalama fiyat: {avg_price:,.0f} TL (Min: {self.min_price_threshold:,} TL filtreli)"

        # En popüler yapılandırma
        if len(budget_laptops) > 0:
            popular_config = budget_laptops.groupby(['ram_gb', 'ssd_gb']).size().idxmax()
            insights[
                'popular_config'] = f"Bu fiyat aralığında en yaygın: {popular_config[0]}GB RAM, {popular_config[1]}GB SSD"

            # GPU önerisi
            if preferences['purpose'] == 'oyun':
                gaming_gpus = budget_laptops[budget_laptops['gpu_score'] >= 70]['gpu_clean'].value_counts()
                if not gaming_gpus.empty:
                    insights['gpu_recommendation'] = f"Oyun için önerilen GPU: {gaming_gpus.index[0].upper()}"

        return insights

    def set_discount_threshold(self, new_threshold: float):
        """Indirim eşiğini değiştir"""
        self.discount_threshold = max(5.0, min(70.0, new_threshold))  # 5-70% arası
        logger.info(f"Fırsat ürün eşiği güncellendi: %{self.discount_threshold}")

    def get_discount_statistics(self, deals_df: pd.DataFrame) -> Dict[str, Any]:
        """İndirim istatistiklerini al"""
        if deals_df.empty:
            return {}
        
        return {
            'discount_distribution': {
                f'{self.discount_threshold}-30%': len(deals_df[deals_df['discount_percentage'] < 30]),
                '30-50%': len(deals_df[(deals_df['discount_percentage'] >= 30) & (deals_df['discount_percentage'] < 50)]),
                '50%+': len(deals_df[deals_df['discount_percentage'] >= 50])
            },
            'top_deals': deals_df.nlargest(5, 'discount_percentage')[['name', 'price', 'market_price', 'discount_percentage']].to_dict('records')
        }


class AdvancedConsoleUI:
    """Gelişmiş kullanıcı arayüzü - Unicode hata düzeltmeli."""

    @staticmethod
    def _get_validated_input(prompt: str, type_conv, validation_fn=lambda x: True, error_msg="Geçersiz giriş.") -> Any:
        while True:
            try:
                value = type_conv(input(prompt))
                if validation_fn(value):
                    return value
                else:
                    print(error_msg)
            except (ValueError, TypeError):
                print("Lütfen geçerli bir değer girin.")
            except KeyboardInterrupt:
                print("\nİşlem iptal edildi.")
                raise

    def get_user_preferences(self) -> Dict[str, Any]:
        """Kullanıcıdan detaylı tercihleri toplar."""
        print("\n" + "=" * 60)
        print("💻 AKILLI LAPTOP ÖNERİ SİSTEMİ 💻".center(60))
        print("=" * 60)

        # Temel bütçe bilgileri
        print("\n📊 BÜTÇE BİLGİLERİ")
        print("-" * 30)
        min_budget = self._get_validated_input(
            "Minimum bütçeniz (TL): ",
            float,
            lambda x: x > 0,
            "Bütçe 0'dan büyük olmalıdır."
        )
        max_budget = self._get_validated_input(
            "Maksimum bütçeniz (TL): ",
            float,
            lambda x: x >= min_budget,
            "Maksimum bütçe, minimumdan küçük olamaz."
        )

        # Kullanım amacı
        print("\n🎯 KULLANIM AMACI")
        print("-" * 30)
        print("1. 🎮 Oyun (Yüksek performans, güçlü GPU)")
        print("2. 🎒 Taşınabilirlik (Hafif, uzun pil ömrü)")
        print("3. 💼 Üretkenlik (Ofis, yazılım geliştirme)")
        print("4. 🎨 Tasarım (Grafik, video düzenleme)")

        purpose_map = {1: 'oyun', 2: 'taşınabilirlik', 3: 'üretkenlik', 4: 'tasarım'}
        purpose_choice = self._get_validated_input(
            "Seçiminiz (1-4): ",
            int,
            lambda x: x in purpose_map,
            "Lütfen 1-4 arası bir seçim yapın."
        )

        # Önem dereceleri
        print("\n⭐ ÖNCELİKLER (1: Önemsiz - 5: Çok Önemli)")
        print("-" * 30)
        prefs = {
            'min_budget': min_budget,
            'max_budget': max_budget,
            'ideal_price': (min_budget + max_budget) / 2,
            'purpose': purpose_map[purpose_choice],
            'performance_importance': self._get_validated_input(
                "Maksimum performans önemi (1-5): ",
                int,
                lambda x: 1 <= x <= 5
            ),
            'battery_importance': self._get_validated_input(
                "Pil ömrü önemi (1-5): ",
                int,
                lambda x: 1 <= x <= 5
            ),
            'portability_importance': self._get_validated_input(
                "Taşınabilirlik önemi (1-5): ",
                int,
                lambda x: 1 <= x <= 5
            ),
        }

        # Gelişmiş filtreleme
        print("\n🔧 GELİŞMİŞ FİLTRELEME (İsteğe Bağlı)")
        print("-" * 30)

        advanced_choice = input("Gelişmiş filtreleme yapmak ister misiniz? (E/H): ").upper()

        if advanced_choice == 'E':
            # Ekran boyutu
            print("\n📐 Ekran Boyutu Tercihi:")
            print("1. Kompakt (13-14\")")
            print("2. Standart (15-16\")")
            print("3. Büyük (17\"+)")
            print("4. Farketmez")
            prefs['screen_preference'] = self._get_validated_input(
                "Seçiminiz (1-4): ",
                int,
                lambda x: 1 <= x <= 4
            )

            # İşletim sistemi
            print("\n💿 İşletim Sistemi Tercihi:")
            print("1. Windows")
            print("2. macOS")
            print("3. Farketmez")
            prefs['os_preference'] = self._get_validated_input(
                "Seçiminiz (1-3): ",
                int,
                lambda x: 1 <= x <= 3
            )

            # Marka tercihi
            brand_pref = input("\n🏢 Tercih ettiğiniz marka (yoksa Enter): ").strip().lower()
            prefs['brand_preference'] = brand_pref if brand_pref else None

            # Minimum donanım
            print("\n⚙️ Minimum Donanım Gereksinimleri:")
            prefs['min_ram'] = self._get_validated_input(
                "Minimum RAM (GB) [Öneri: 8]: ",
                int,
                lambda x: x >= 4,
                "En az 4GB RAM seçmelisiniz."
            )
            prefs['min_ssd'] = self._get_validated_input(
                "Minimum SSD (GB) [Öneri: 256]: ",
                int,
                lambda x: x >= 128,
                "En az 128GB SSD seçmelisiniz."
            )
        else:
            # Varsayılan değerler
            prefs['screen_preference'] = 4
            prefs['os_preference'] = 3
            prefs['brand_preference'] = None
            prefs['min_ram'] = 8
            prefs['min_ssd'] = 256

        return prefs

    def display_recommendations(self, recommendations: pd.DataFrame, preferences: Dict[str, Any],
                                trends: Dict[str, Any] = None, insights: Dict[str, str] = None):
        """Önerileri detaylı ve görsel olarak göster."""
        if recommendations.empty:
            print("\n" + "=" * 60)
            print("❌ SONUÇ BULUNAMADI ❌".center(60))
            print("=" * 60)
            print("\nBelirttiğiniz kriterlere uygun laptop bulunamadı.")
            print("\nÖneriler:")
            print("• Bütçenizi artırmayı deneyin")
            print("• Minimum donanım gereksinimlerini azaltın")
            print("• Marka tercihini kaldırın")
            return

        print("\n" + "=" * 60)
        print(f"✅ SİZE ÖZEL {len(recommendations)} LAPTOP ÖNERİSİ ✅".center(60))
        print("=" * 60)

        # Pazar içgörüleri
        if insights:
            print("\n📊 PAZAR ANALİZİ")
            print("-" * 60)
            for key, value in insights.items():
                print(f"• {value}")

        # En iyi alım zamanı önerisi
        if trends:
            analyzer = TrendAnalyzer()
            best_time = analyzer.suggest_best_time_to_buy(preferences['purpose'])
            print(f"\n📅 En iyi alım zamanı: {best_time}")

        print("\n" + "=" * 60)
        print("🏆 ÖNERİLER".center(60))
        print("=" * 60)

        for i, (_, laptop) in enumerate(recommendations.iterrows(), 1):
            # Başlık
            print(f"\n{i}. {laptop['name']}")
            print("─" * 60)

            # Temel bilgiler
            source_icon = "🛒" if '2' in laptop['data_source'] else "📊"
            print(f"{source_icon} Fiyat: {laptop['price']:,.0f} TL | Puan: {laptop['score']:.1f}/100")

            # Donanım detayları
            print(f"\n📋 Teknik Özellikler:")
            print(f"   • Ekran: {laptop['screen_size']}\"")
            print(f"   • İşlemci: {laptop['cpu_clean'].upper()}")
            print(f"   • Grafik: {laptop['gpu_clean'].upper()}")
            print(f"   • RAM: {int(laptop['ram_gb'])} GB")
            print(f"   • Depolama: {int(laptop['ssd_gb'])} GB SSD")
            print(f"   • İşletim Sistemi: {laptop['os']}")
            print(f"   • Marka: {laptop['brand'].title()}")

            # Öne çıkan özellikler
            features = self._get_laptop_features(laptop)
            if features:
                print(f"\n✨ Öne Çıkan Özellikler:")
                for feature in features:
                    print(f"   {feature}")

            # Öneri açıklaması
            explanation = self._generate_recommendation_explanation(laptop, preferences)
            print(f"\n💡 Neden bu laptop?")
            print(f"   {explanation}")

            # Uyarılar
            warnings = self._get_laptop_warnings(laptop)
            if warnings:
                print(f"\n⚠️  Dikkat edilmesi gerekenler:")
                for warning in warnings:
                    print(f"   {warning}")

            # Link
            print(f"\n🔗 Ürün Linki: {laptop['url']}")
            print("─" * 60)

        # Karşılaştırma tablosu
        if len(recommendations) > 1:
            print("\n📊 KARŞILAŞTIRMA TABLOSU")
            print("=" * 60)
            print(self._create_comparison_table(recommendations))

    def _get_laptop_features(self, laptop: pd.Series) -> List[str]:
        """Laptop'un öne çıkan özelliklerini belirle"""
        features = []

        if laptop['is_apple']:
            features.append("🍎 Apple ekosistemi ile tam uyum")
        if laptop['has_dedicated_gpu'] and laptop['gpu_score'] >= 80:
            features.append("🚀 Üst düzey oyun performansı")
        elif laptop['has_dedicated_gpu']:
            features.append("🎮 Oyun ve grafik işlemleri için uygun")
        else:
            features.append("🔋 Uzun pil ömrü için optimize edilmiş")

        if laptop['ram_gb'] >= 32:
            features.append("💪 Profesyonel işler için güçlü bellek")
        elif laptop['ram_gb'] >= 16:
            features.append("⚡ Çoklu görevler için ideal bellek")

        if laptop['ssd_gb'] >= 1000:
            features.append("💾 Geniş depolama alanı")

        if laptop['screen_size'] <= 14:
            features.append("🎒 Kompakt ve taşınabilir tasarım")
        elif laptop['screen_size'] >= 17:
            features.append("🖥️ Geniş ekran deneyimi")

        if laptop['brand_score'] >= 0.85:
            features.append("⭐ Yüksek marka güvenilirliği")

        return features

    def _generate_recommendation_explanation(self, laptop: pd.Series, preferences: Dict[str, Any]) -> str:
        """Öneri için detaylı açıklama üret"""
        explanations = []

        # Fiyat açıklaması
        price_ratio = laptop['price'] / preferences['ideal_price']
        if price_ratio < 0.9:
            explanations.append("Bütçenizin altında ekonomik bir seçim")
        elif price_ratio > 1.1:
            explanations.append("Bütçenizin biraz üzerinde ama sunduğu özellikler değerli")
        else:
            explanations.append("Tam bütçenize uygun")

        # Kullanım amacına uygunluk
        purpose_fit = {
            'oyun': "oyun performansı" if laptop['gpu_score'] > 70 else "casual oyunlar",
            'taşınabilirlik': "mobilite" if laptop['screen_size'] <= 14 or not laptop[
                'has_dedicated_gpu'] else "dengeli taşınabilirlik",
            'üretkenlik': "verimli çalışma",
            'tasarım': "yaratıcı projeler"
        }
        explanations.append(f"{purpose_fit[preferences['purpose']]} için optimize")

        # Fiyat/performans
        perf_score = (laptop['gpu_score'] + laptop['cpu_score']) / 2
        if perf_score > 80:
            explanations.append("mükemmel fiyat/performans oranı")
        elif perf_score > 60:
            explanations.append("iyi fiyat/performans dengesi")

        return ", ".join(explanations) + "."

    def _get_laptop_warnings(self, laptop: pd.Series) -> List[str]:
        """Potansiyel uyarıları belirle"""
        warnings = []

        if laptop['is_price_outlier']:
            warnings.append("• Fiyat ortalamanın çok dışında, dikkatli olun")
        if laptop['ram_gb'] < 8:
            warnings.append("• RAM modern uygulamalar için düşük olabilir")
        if laptop['ssd_gb'] < 256:
            warnings.append("• Depolama alanı sınırlı olabilir")
        if not laptop['has_dedicated_gpu'] and laptop['brand'] != 'apple':
            warnings.append("• Harici grafik kartı yok, ağır oyunlar için uygun değil")
        if laptop['screen_size'] >= 17:
            warnings.append("• Büyük boyut nedeniyle taşınabilirlik sınırlı")

        return warnings

    def _create_comparison_table(self, recommendations: pd.DataFrame) -> str:
        """Karşılaştırma tablosu oluştur"""
        comparison_data = []

        for idx, (_, laptop) in enumerate(recommendations.iterrows(), 1):
            name_short = laptop['name'][:25] + '...' if len(laptop['name']) > 25 else laptop['name']
            comparison_data.append([
                idx,
                name_short,
                f"{laptop['price']:,.0f}",
                f"{laptop['screen_size']}\"",
                f"{int(laptop['ram_gb'])}GB",
                f"{int(laptop['ssd_gb'])}GB",
                laptop['gpu_clean'][:8].upper(),
                f"{laptop['score']:.1f}"
            ])

        headers = ['#', 'Model', 'Fiyat (TL)', 'Ekran', 'RAM', 'SSD', 'GPU', 'Puan']
        return tabulate(comparison_data, headers=headers, tablefmt='grid')

    def display_deals(self, deals_df: pd.DataFrame, insights: Dict[str, Any]):
        """Fırsat ürünlerini görüntüle - Hata düzeltmeli versiyon"""
        if deals_df.empty:
            print("\n" + "=" * 60)
            print("💔 FIRSAT ÜRÜN BULUNAMADI 💔".center(60))
            print("=" * 60)
            print("\nŞu anda öne çıkan fırsat ürün bulunmuyor.")
            print("Daha sonra tekrar kontrol edebilirsiniz.")
            return

        print("\n" + "=" * 80)
        print("🎯 PIYASA FIYATINA GÖRE FIRSAT ÜRÜNLERİ 🎯".center(80))
        print("=" * 80)

        # Genel içgörüler
        if insights:
            print("\n📊 FIRSAT ANALİZİ")
            print("-" * 80)
            print(f"• Toplam {insights['total_deals']} fırsat ürün tespit edildi")
            print(f"• Ortalama indirim oranı: %{insights['avg_discount']:.1f}")
            print(f"• Maksimum indirim: %{insights['max_discount']:.1f}")
            print(f"• En çok fırsat sunan kategori: {insights['best_deal_category'].upper()}")
            print(f"• Fiyat aralığı: {insights['price_range'][0]:,.0f} - {insights['price_range'][1]:,.0f} TL")
            if insights['top_brands']:
                print(f"• En çok fırsat sunan markalar: {', '.join([f'{k.title()} ({v})' for k, v in insights['top_brands'].items()])}")
            
            # Tasarruf bilgileri
            if 'savings_info' in insights:
                savings = insights['savings_info']
                print(f"• Toplam tasarruf potansiyeli: {savings['total_savings']:,.0f} TL")
                print(f"• Ortalama tasarruf: {savings['avg_savings']:,.0f} TL")

        print("\n" + "=" * 80)
        print("🏆 EN İYİ FIRSATLAR".center(80))
        print("=" * 80)

        # En iyi 10 fırsatı göster
        top_deals = deals_df.head(10)

        for i, (_, deal) in enumerate(top_deals.iterrows(), 1):
            # Fırsat seviyesi belirleme
            if deal['deal_score'] >= 80:
                deal_level = "🔥 MUHTEŞEM FIRSAT"
                level_color = "🟥"
            elif deal['deal_score'] >= 60:
                deal_level = "⭐ ÇOK İYİ FIRSAT"
                level_color = "🟧"
            else:
                deal_level = "✨ İYİ FIRSAT"
                level_color = "🟨"

            print(f"\n{i}. {deal['name']}")
            print("─" * 80)
            print(f"{level_color} {deal_level} | Fırsat Skoru: {deal['deal_score']:.1f}/100")
            
            # Fiyat bilgileri - güvenli erişim
            market_price = deal.get('market_price', deal['price'] * 1.3)  # Eğer yoksa %30 ekle
            discount_pct = deal.get('discount_percentage', 0)
            
            print(f"💰 Fiyat: {deal['price']:,.0f} TL (Piyasa ort: {market_price:,.0f} TL)")
            print(f"📉 İndirim Oranı: %{discount_pct:.1f}")

            # Özellikler
            print(f"\n📋 Özellikler:")
            print(f"   • İşlemci: {deal['cpu_clean'].upper()} (Skor: {deal['cpu_score']}/100)")
            print(f"   • Grafik: {deal['gpu_clean'].upper()} (Skor: {deal['gpu_score']}/100)")
            print(f"   • RAM: {int(deal['ram_gb'])} GB | SSD: {int(deal['ssd_gb'])} GB")
            print(f"   • Ekran: {deal['screen_size']}\" | Marka: {deal.get('brand', 'Bilinmiyor').title()}")

            # Neden fırsat?
            print(f"\n💡 Neden fırsat?")
            reasons = []

            if discount_pct > 20:
                reasons.append(f"Piyasa fiyatından %{discount_pct:.0f} daha ucuz")

            # Performans değeri kontrolü
            if 'performance_score' in deal:
                perf_value = deal['performance_score'] / (deal['price'] / 10000)
                if perf_value > 5:  # Eşik değeri
                    reasons.append("Fiyat/performans oranı çok yüksek")

            if deal['gpu_score'] >= 70 and deal['price'] < 40000:
                reasons.append("Oyun performansı için uygun fiyat")

            if deal['ram_gb'] >= 16 and deal['ssd_gb'] >= 512:
                reasons.append("Yüksek bellek ve depolama kapasitesi")
            
            if deal['deal_score'] >= 70:
                reasons.append("Genel değerlendirmede yüksek skor")

            for reason in reasons:
                print(f"   ✓ {reason}")

            # Link
            print(f"\n🔗 Ürün Linki: {deal['url']}")
            print("─" * 80)

        # Ek öneriler
        print("\n💡 ÖNERİLER")
        print("-" * 80)
        print("• Bu fırsatlar piyasa analizi sonucu tespit edilmiştir")
        print("• Satın almadan önce ürün incelemelerini okuyun")
        print("• Satıcı güvenilirliğini mutlaka kontrol edin")
        print("• Garanti ve servis koşullarını araştırın")


class RobustRecommender:
    """Ana uygulama sınıfı - Tüm bileşenleri yönetir - DÜZELTME."""

    def __init__(self):
        self.config = Config()
        self.cached_data_handler = CachedDataHandler(self.config)
        self.data_handler = EnhancedDataHandler(self.config)
        self.scoring_engine = AdvancedScoringEngine(self.config)
        self.ml_recommender = MLRecommender()
        self.trend_analyzer = TrendAnalyzer()
        self.ui = AdvancedConsoleUI()
        self.df = None
        self.cleaning_stats = None
        self.setup_advanced_logging()

    def setup_advanced_logging(self):
        """Unicode güvenli loglama sistemi"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Log dosyası için tarih damgası
        log_filename = f'laptop_recommender_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        # Emoji temizleyici formatter
        class SafeFormatter(logging.Formatter):
            def format(self, record):
                msg = super().format(record)
                # Unicode emoji'leri ASCII'ye çevir
                emoji_map = {
                    '✅': '[OK]',
                    '❌': '[ERROR]',
                    '⚠️': '[WARNING]',
                    '🔧': '[PROCESSING]',
                    '📊': '[DATA]',
                    '🚀': '[START]',
                    '💻': '[LAPTOP]',
                    '🎯': '[TARGET]'
                }
                for emoji, text in emoji_map.items():
                    msg = msg.replace(emoji, text)
                return msg
        
        # UTF-8 ile dosya handler
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        console_handler = logging.StreamHandler()
        
        file_handler.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.INFO)
        
        safe_formatter = SafeFormatter(log_format)
        console_handler.setFormatter(safe_formatter)
        
        # Dosya için normal formatter
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        
        # Root logger'a handler'ları ekle
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    def setup(self):
        """Veriyi yükler ve temizler."""
        try:
            print("\n⏳ Veriler yükleniyor...")
            raw_df = self.cached_data_handler.load_and_merge_data()

            print("🔧 Veriler temizleniyor ve işleniyor...")
            self.df, self.cleaning_stats = self.data_handler.clean_data(raw_df)

            print(f"✅ {len(self.df)} laptop başarıyla yüklendi!")

            # Temizleme istatistiklerini göster
            if self.cleaning_stats['outliers_detected'] > 0:
                print(f"⚠️  {self.cleaning_stats['outliers_detected']} aykırı fiyat tespit edildi")
            if self.cleaning_stats['suspicious_items'] > 0:
                print(f"⚠️  {self.cleaning_stats['suspicious_items']} şüpheli ürün tespit edildi")

        except Exception as e:
            logger.error(f"Veri yükleme hatası: {e}")
            raise

    def debug_scoring_system(self):
        """Puanlama sistemini test et"""
        print("🔧 Puanlama sistemi test ediliyor...")
        
        # Basit test preferences
        test_prefs = {
            'min_budget': 20000,
            'max_budget': 50000,
            'ideal_price': 35000,
            'purpose': 'oyun',
            'performance_importance': 5,
            'battery_importance': 2,
            'portability_importance': 2,
            'screen_preference': 4,
            'os_preference': 3,
            'brand_preference': None,
            'min_ram': 8,
            'min_ssd': 256
        }
        
        # İlk 5 laptop ile test et
        test_df = self.df.head(5).copy()
        print(f"Test dataframe: {len(test_df)} laptop")
        
        scored_test = self.scoring_engine.calculate_scores_parallel(test_df, test_prefs)
        
        if not scored_test.empty:
            print("✅ Test başarılı!")
            print(f"Score tipi: {scored_test['score'].dtype}")
            print(f"Score değerleri: {scored_test['score'].tolist()}")
            return True
        else:
            print("❌ Test başarısız!")
            return False

    def get_recommendations(self, preferences: Dict[str, Any]) -> pd.DataFrame:
        """Kullanıcı tercihlerine göre öneriler üret - DÜZELTME"""
        # Puanlama yap
        print("\n🧮 Laptoplar analiz ediliyor...")
        scored_df = self.scoring_engine.calculate_scores_parallel(self.df, preferences)

        if scored_df.empty:
            print("❌ Filtreleme sonrası hiç laptop bulunamadı!")
            return pd.DataFrame()

        # DEBUG: Score sütunu kontrol et
        print(f"Score sütunu tipi: {scored_df['score'].dtype}")
        
        # Score sütununun numeric olduğundan emin ol
        if scored_df['score'].dtype == 'object':
            print("⚠️ Score sütunu object tipinde, düzeltiliyor...")
            scored_df['score'] = pd.to_numeric(scored_df['score'], errors='coerce')
            scored_df['score'] = scored_df['score'].fillna(0.0)
            print(f"Düzeltme sonrası tip: {scored_df['score'].dtype}")

        # En yüksek puanlı laptopları al
        try:
            top_laptops = scored_df.nlargest(10, 'score')
            print(f"✅ En iyi {len(top_laptops)} laptop seçildi")
        except Exception as e:
            print(f"❌ nlargest hatası: {e}")
            # Alternatif yöntem
            scored_df_sorted = scored_df.sort_values('score', ascending=False)
            top_laptops = scored_df_sorted.head(10)
            print(f"✅ Alternatif yöntemle {len(top_laptops)} laptop seçildi")

        # ML kısmını geçici olarak devre dışı bırak
        # if len(top_laptops) > 0 and preferences.get('use_ml', True):
        #     try:
        #         print("🤖 Makine öğrenmesi ile ek öneriler aranıyor...")
        #         # ML kodları...
        #     except Exception as e:
        #         logger.warning(f"ML önerisi başarısız: {e}")

        # Final olarak en iyi 5'i döndür
        final_recommendations = top_laptops.head(5)
        print(f"📋 {len(final_recommendations)} öneri hazırlandı")
        
        return final_recommendations

    def run_with_recovery(self):
        """Hata durumunda kurtarma mekanizmalı çalıştırma - DEBUG eklendi"""
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Veriyi yükle
                self.setup()
                
                # DEBUG: Puanlama sistemini test et
                if not self.debug_scoring_system():
                    print("⚠️ Puanlama sistemi test edilemiyor, devam ediliyor...")

                # Ana menü
                print("\n" + "=" * 60)
                print("🏠 ANA MENÜ")
                print("=" * 60)
                print("1. Kişiselleştirilmiş laptop önerisi al")
                print("2. 🎯 Günün fırsat ürünlerini gör")
                print("3. Çıkış")

                main_choice = input("\nSeçiminiz (1-3): ")

                if main_choice == '1':
                    # Mevcut öneri akışı
                    preferences = self.ui.get_user_preferences()
                    recommendations = self.get_recommendations(preferences)
                    trends = self.trend_analyzer.analyze_price_trends(self.df)
                    insights = self.trend_analyzer.get_market_insights(self.df, preferences)
                    self.ui.display_recommendations(recommendations, preferences, trends, insights)
                    self._offer_additional_options(recommendations, preferences)

                elif main_choice == '2':
                    # Direkt fırsat ürünleri
                    self._show_deal_products()

                    # Tekrar ana menüye dön
                    continue_choice = input("\n\nAna menüye dönmek ister misiniz? (E/H): ").upper()
                    if continue_choice == 'E':
                        continue
                    else:
                        break

                elif main_choice == '3':
                    print("\n👋 İyi günler!")
                    break
                else:
                    print("\nGeçersiz seçim.")
                    continue

                break

            except KeyboardInterrupt:
                print("\n\n👋 Program sonlandırılıyor. İyi günler!")
                break

            except Exception as e:
                retry_count += 1
                logger.error(f"Hata oluştu (Deneme {retry_count}/{max_retries}): {e}")

                if retry_count < max_retries:
                    print(f"\n⚠️ Bir hata oluştu. Yeniden deneniyor... ({retry_count}/{max_retries})")
                    time.sleep(2)
                else:
                    print("\n❌ Program başarısız oldu.")
                    print(f"Hata detayı: {e}")
                    print("Lütfen log dosyasını kontrol edin.")
                    raise

    def _offer_additional_options(self, recommendations: pd.DataFrame, preferences: Dict[str, Any]):
        """Kullanıcıya ek seçenekler sun"""
        print("\n" + "=" * 60)
        print("📌 EK SEÇENEKLER")
        print("=" * 60)
        print("1. Yeni arama yap")
        print("2. Detaylı karşılaştırma raporu oluştur")
        print("3. Benzer ürünleri göster")
        print("4. 🎯 Fırsat ürünlerini göster (Anomali tespiti)")
        print("5. Çıkış")

        choice = input("\nSeçiminiz (1-5): ")

        if choice == '1':
            self.run_with_recovery()
        elif choice == '2':
            self._generate_detailed_report(recommendations, preferences)
        elif choice == '3':
            self._show_similar_products(recommendations)
        elif choice == '4':
            self._show_deal_products()
        elif choice == '5':
            print("\n👋 İyi günler!")
        else:
            print("\nGeçersiz seçim. Program sonlandırılıyor.")

    def _show_deal_products(self):
        """Fırsat ürünlerini göster"""
        print("\n🔍 Fırsat ürünleri analiz ediliyor...")
        print("(Anomali tespiti algoritması çalışıyor, lütfen bekleyin...)")

        try:
            # Fırsatları bul
            deals = self.trend_analyzer.find_deal_products(self.df)

            if not deals.empty:
                # İçgörüleri al
                insights = self.trend_analyzer.get_deal_insights(deals)

                # Görüntüle
                self.ui.display_deals(deals, insights)

                # Filtreleme seçeneği sun
                print("\n" + "=" * 60)
                filter_choice = input("\nBelirli bir bütçe aralığında fırsat görmek ister misiniz? (E/H): ").upper()

                if filter_choice == 'E':
                    min_price = float(input("Minimum fiyat (TL): "))
                    max_price = float(input("Maksimum fiyat (TL): "))

                    filtered_deals = deals[
                        (deals['price'] >= min_price) &
                        (deals['price'] <= max_price)
                        ]

                    if not filtered_deals.empty:
                        print(f"\n📌 {min_price:,.0f} - {max_price:,.0f} TL aralığındaki fırsatlar:")
                        insights_filtered = self.trend_analyzer.get_deal_insights(filtered_deals)
                        self.ui.display_deals(filtered_deals, insights_filtered)
                    else:
                        print(f"\n❌ Bu fiyat aralığında fırsat ürün bulunamadı.")
            else:
                self.ui.display_deals(pd.DataFrame(), {})

        except Exception as e:
            logger.error(f"Fırsat ürün analizi hatası: {e}")
            print(f"\n❌ Fırsat analizi sırasında hata oluştu: {e}")

    def _generate_detailed_report(self, recommendations: pd.DataFrame, preferences: Dict[str, Any]):
        """Detaylı rapor oluştur"""
        report_filename = f"laptop_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("LAPTOP ÖNERİ RAPORU\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Oluşturulma Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("KULLANICI TERCİHLERİ\n")
            f.write("-" * 60 + "\n")
            for key, value in preferences.items():
                f.write(f"{key}: {value}\n")

            f.write("\n\nÖNERİLEN LAPTOPLAR\n")
            f.write("-" * 60 + "\n")
            for i, (_, laptop) in enumerate(recommendations.iterrows(), 1):
                f.write(f"\n{i}. {laptop['name']}\n")
                f.write(f"   Fiyat: {laptop['price']:,.0f} TL\n")
                f.write(f"   Puan: {laptop['score']:.1f}/100\n")
                f.write(f"   URL: {laptop['url']}\n")

        print(f"\n✅ Detaylı rapor '{report_filename}' dosyasına kaydedildi.")

    def _show_similar_products(self, recommendations: pd.DataFrame):
        """Benzer ürünleri göster"""
        if recommendations.empty:
            print("\nÖneri bulunamadığı için benzer ürün gösterilemiyor.")
            return

        print("\n🔍 İlk öneriye benzer ürünler aranıyor...")

        try:
            # İlk öneriyi al
            first_recommendation = recommendations.iloc[0]
            first_idx = self.df[self.df['name'] == first_recommendation['name']].index[0]

            # Benzer ürünleri bul
            similar = self.ml_recommender.find_similar_laptops(first_idx, self.df, n_recommendations=5)

            print(f"\n📋 '{first_recommendation['name']}' modeline benzer ürünler:")
            print("=" * 60)

            for i, (_, laptop) in enumerate(similar.iterrows(), 1):
                print(f"{i}. {laptop['name']}")
                print(f"   Fiyat: {laptop['price']:,.0f} TL")
                print(f"   Benzerlik: {laptop['similarity_score']:.2%}")
                print()

        except Exception as e:
            logger.error(f"Benzer ürün arama hatası: {e}")
            print("\n❌ Benzer ürünler bulunamadı.")


# Ana program
if __name__ == "__main__":
    import sys

    # Gerekli kütüphaneleri kontrol et
    required_packages = ['pandas', 'numpy', 'sklearn', 'tabulate']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ Eksik kütüphaneler: {', '.join(missing_packages)}")
        print(f"Lütfen şu komutu çalıştırın: pip install {' '.join(missing_packages)}")
        sys.exit(1)

    # Uygulamayı başlat
    print("\n" + "=" * 60)
    print("🚀 LAPTOP ÖNERİ SİSTEMİ BAŞLATILIYOR...")
    print("=" * 60)

    recommender = RobustRecommender()

    # Test modunu kontrol et
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        print("\n🧪 Test modu aktif...")
        # Test kodları eklenebilir
        import unittest


        class TestEnhancedSystem(unittest.TestCase):
            """Geliştirilmiş sistem için birim testler."""

            def setUp(self):
                """Test verisi hazırla."""
                self.config = Config()
                self.data_handler = EnhancedDataHandler(self.config)
                self.scoring_engine = AdvancedScoringEngine(self.config)

                # Test dataframe'i oluştur
                self.test_df = pd.DataFrame({
                    'url': ['https://example.com/1', 'https://example.com/2'],
                    'name': ['ASUS Gaming Laptop', 'Apple MacBook Pro'],
                    'brand': ['asus', 'apple'],
                    'price': [25000, 45000],
                    'screen_size': [15.6, 14.0],
                    'ssd_gb': [512, 1024],
                    'ram_gb': [16, 16],
                    'os': ['WINDOWS 11', 'MACOS'],
                    'gpu_clean': ['rtx4060', 'apple integrated'],
                    'cpu_clean': ['i7', 'm3'],
                    'gpu_score': [75, 35],
                    'cpu_score': [85, 85],
                    'brand_score': [0.82, 0.95],
                    'is_apple': [False, True],
                    'has_dedicated_gpu': [True, False],
                    'is_price_outlier': [False, False],
                    'is_suspicious': [False, False],
                    'data_source': ['dataset_1', 'dataset_1']
                })

            def test_advanced_scoring(self):
                """Gelişmiş puanlama sistemini test et"""
                preferences = {
                    'min_budget': 20000,
                    'max_budget': 50000,
                    'ideal_price': 35000,
                    'purpose': 'oyun',
                    'performance_importance': 5,
                    'battery_importance': 2,
                    'portability_importance': 2,
                    'screen_preference': 4,
                    'os_preference': 3,
                    'brand_preference': None,
                    'min_ram': 8,
                    'min_ssd': 256
                }

                scored_df = self.scoring_engine.calculate_scores_parallel(
                    self.test_df, preferences
                )

                # Oyun için ASUS'un daha yüksek puan alması gerekir
                self.assertGreater(
                    scored_df[scored_df['name'].str.contains('ASUS')]['score'].iloc[0],
                    scored_df[scored_df['name'].str.contains('Apple')]['score'].iloc[0]
                )

            def test_ml_recommender(self):
                """ML öneri sistemini test et"""
                ml_rec = MLRecommender()

                # Daha büyük test verisi oluştur
                large_test_df = pd.concat([self.test_df] * 10, ignore_index=True)

                # İlk laptop'a benzer olanları bul
                similar = ml_rec.find_similar_laptops(0, large_test_df, n_recommendations=3)

                self.assertEqual(len(similar), 3)
                self.assertTrue('similarity_score' in similar.columns)

            def test_trend_analyzer(self):
                """Trend analizini test et"""
                analyzer = TrendAnalyzer()
                trends = analyzer.analyze_price_trends(self.test_df)

                self.assertIn('gpu_prices', trends)
                self.assertIn('brand_prices', trends)
                self.assertIn('price_distribution', trends)

            def test_filter_application(self):
                """Filtreleme sistemini test et"""
                # Sadece Windows filtresi
                preferences = {
                    'min_budget': 10000,
                    'max_budget': 100000,
                    'ideal_price': 50000,
                    'purpose': 'üretkenlik',
                    'os_preference': 1,  # Windows
                    'screen_preference': 4,
                    'brand_preference': None,
                    'min_ram': 8,
                    'min_ssd': 256
                }

                filtered_df = self.scoring_engine._apply_filters(
                    self.test_df, preferences
                )

                # Sadece Windows laptop kalmalı
                self.assertEqual(len(filtered_df), 1)
                self.assertTrue(filtered_df['os'].str.contains('WINDOWS').all())


        # Testleri çalıştır
        unittest.main(argv=[''], exit=False, verbosity=2)

    else:
        # Normal modda çalıştır
        try:
            recommender.run_with_recovery()
        except Exception as e:
            logger.critical(f"Kritik hata: {e}")
            print(f"\n💔 Beklenmeyen bir hata oluştu: {e}")
            print("Lütfen log dosyasını kontrol edin veya geliştiriciyle iletişime geçin.")

            sys.exit(1)
