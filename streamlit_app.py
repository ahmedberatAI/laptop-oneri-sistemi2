import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# laptop_engine.py dosyasını import et
from laptop_engine import (
    Config, 
    CachedDataHandler, 
    EnhancedDataHandler,
    AdvancedScoringEngine, 
    TrendAnalyzer,
    RobustRecommender
)

# Streamlit sayfa ayarları
st.set_page_config(
    page_title="💻 Laptop Öneri Sistemi",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS stilleri
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .deal-card {
        background: linear-gradient(135deg, #ff6b6b, #feca57);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Ana başlık
st.markdown('<h1 class="main-header">💻 Akıllı Laptop Öneri Sistemi</h1>', unsafe_allow_html=True)

# Veri yükleme (cache ile)
@st.cache_data
def load_system():
    """Sistem bileşenlerini yükle"""
    try:
        recommender = RobustRecommender()
        recommender.setup()
        return recommender
    except Exception as e:
        st.error(f"Sistem yüklenemedi: {e}")
        return None

# Ana uygulama
def main():
    # Sistem yükleme
    with st.spinner("💻 Sistem başlatılıyor..."):
        recommender = load_system()
    
    if not recommender:
        st.error("❌ Sistem başlatılamadı!")
        return
    
    # Sidebar - Navigasyon
    page = st.sidebar.selectbox(
        "📱 Sayfa Seçin",
        ["🏠 Ana Sayfa", "🔍 Laptop Önerileri", "🔥 Günün Fırsatları", "📊 Pazar Analizi"]
    )
    
    if page == "🏠 Ana Sayfa":
        show_home_page(recommender)
    elif page == "🔍 Laptop Önerileri":
        show_recommendations_page(recommender)
    elif page == "🔥 Günün Fırsatları":
        show_deals_page(recommender)
    elif page == "📊 Pazar Analizi":
        show_analytics_page(recommender)

def show_home_page(recommender):
    """Ana sayfa"""
    col1, col2, col3 = st.columns(3)
    
    total_laptops = len(recommender.df)
    avg_price = recommender.df['price'].mean()
    brands = recommender.df['brand'].nunique()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📱 Toplam Laptop</h3>
            <h2>{total_laptops:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💰 Ortalama Fiyat</h3>
            <h2>{avg_price:,.0f} ₺</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏢 Toplam Marka</h3>
            <h2>{brands}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Hızlı istatistikler
    st.subheader("📊 Hızlı İstatistikler")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fiyat dağılımı
        fig_price = px.histogram(
            recommender.df, 
            x='price', 
            nbins=20,
            title="💰 Fiyat Dağılımı",
            color_discrete_sequence=['#667eea']
        )
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        # Marka dağılımı
        brand_counts = recommender.df['brand'].value_counts().head(8)
        fig_brand = px.bar(
            x=brand_counts.index,
            y=brand_counts.values,
            title="🏢 Popüler Markalar",
            color_discrete_sequence=['#764ba2']
        )
        st.plotly_chart(fig_brand, use_container_width=True)

def show_recommendations_page(recommender):
    """Öneri sayfası"""
    st.header("🔍 Size Özel Laptop Önerileri")
    
    # Sidebar - Tercihler
    with st.sidebar:
        st.subheader("🎯 Tercihlerinizi Belirtin")
        
        # Bütçe
        budget_range = st.slider(
            "💰 Bütçe Aralığı (₺)",
            min_value=5000,
            max_value=150000,
            value=(20000, 80000),
            step=5000
        )
        
        # Kullanım amacı
        purpose = st.selectbox(
            "🎮 Kullanım Amacı",
            ['oyun', 'taşınabilirlik', 'üretkenlik', 'tasarım'],
            format_func=lambda x: {
                'oyun': '🎮 Oyun',
                'taşınabilirlik': '🎒 Taşınabilirlik',
                'üretkenlik': '💼 Üretkenlik', 
                'tasarım': '🎨 Tasarım'
            }[x]
        )
        
        # Önem dereceleri
        st.subheader("⭐ Önemli Faktörler")
        performance = st.slider("🚀 Performans", 1, 5, 4)
        battery = st.slider("🔋 Pil Ömrü", 1, 5, 3)
        portability = st.slider("🎒 Taşınabilirlik", 1, 5, 3)
        
        # Gelişmiş filtreler
        with st.expander("⚙️ Gelişmiş Ayarlar"):
            min_ram = st.selectbox("💾 Min RAM (GB)", [4, 8, 16, 32], index=1)
            min_ssd = st.selectbox("💿 Min SSD (GB)", [128, 256, 512, 1024], index=1)
            
            brand_pref = st.selectbox(
                "🏢 Tercih Edilen Marka",
                [''] + list(recommender.df['brand'].unique()),
                format_func=lambda x: 'Farketmez' if x == '' else x.title()
            )
    
    # Öneri butonu
    if st.button("🔍 Laptop Önerilerini Getir", type="primary", use_container_width=True):
        
        # Tercihleri hazırla
        preferences = {
            'min_budget': budget_range[0],
            'max_budget': budget_range[1], 
            'ideal_price': sum(budget_range) / 2,
            'purpose': purpose,
            'performance_importance': performance,
            'battery_importance': battery,
            'portability_importance': portability,
            'screen_preference': 4,  # Farketmez
            'os_preference': 3,      # Farketmez
            'brand_preference': brand_pref if brand_pref else None,
            'min_ram': min_ram,
            'min_ssd': min_ssd
        }
        
        with st.spinner("🤖 AI size özel laptopları buluyor..."):
            recommendations = recommender.get_recommendations(preferences)
        
        if not recommendations.empty:
            st.success(f"✅ {len(recommendations)} mükemmel öneri bulundu!")
            
            # Önerileri göster
            for i, (_, laptop) in enumerate(recommendations.iterrows(), 1):
                with st.container():
                    st.markdown(f"### {i}. {laptop['name']}")
                    
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"**💰 Fiyat:** {laptop['price']:,} ₺")
                        st.markdown(f"**🏢 Marka:** {laptop['brand'].title()}")
                        st.markdown(f"**💾 RAM:** {int(laptop['ram_gb'])} GB | **💿 SSD:** {int(laptop['ssd_gb'])} GB")
                        st.markdown(f"**📱 Ekran:** {laptop['screen_size']}\" | **🎮 GPU:** {laptop['gpu_clean'].upper()}")
                    
                    with col2:
                        # Skor göstergesi
                        score = laptop['score']
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = score,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Uygunluk"},
                            gauge = {'axis': {'range': [None, 100]},
                                   'bar': {'color': "darkgreen"},
                                   'steps': [
                                       {'range': [0, 50], 'color': "lightgray"},
                                       {'range': [50, 80], 'color': "yellow"}],
                                   'threshold': {'line': {'color': "red", 'width': 4},
                                               'thickness': 0.75, 'value': 90}}))
                        fig.update_layout(height=200)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col3:
                        st.markdown("**✨ Öne Çıkan Özellikler:**")
                        
                        features = []
                        if laptop['gpu_score'] >= 70:
                            features.append("🎮 Güçlü GPU")
                        if laptop['ram_gb'] >= 16:
                            features.append("⚡ Yüksek RAM")
                        if laptop['brand_score'] >= 0.85:
                            features.append("⭐ Güvenilir marka")
                        if laptop['ssd_gb'] >= 512:
                            features.append("💿 Geniş depolama")
                        
                        for feature in features[:3]:
                            st.markdown(f"• {feature}")
                    
                    st.markdown(f"🔗 [Ürünü İncele]({laptop['url']})")
                    st.divider()
        else:
            st.warning("❗ Kriterlere uygun laptop bulunamadı. Filtrelerinizi genişletmeyi deneyin.")

def show_deals_page(recommender):
    """Fırsat ürünleri sayfası"""
    st.header("🔥 Günün En İyi Fırsatları")
    
    # Filtreleme seçenekleri
    col1, col2, col3 = st.columns(3)
    with col1:
        show_count = st.selectbox("📱 Gösterilecek Fırsat Sayısı", [10, 20, 30, 50], index=1)
    with col2:
        min_discount = st.slider("📉 Minimum İndirim %", 5, 50, 15)
    with col3:
        max_price = st.number_input("💰 Maksimum Fiyat (₺)", min_value=5000, max_value=200000, value=100000, step=5000)
    
    with st.spinner("🔍 Piyasa analizi yapılıyor..."):
        # İndirim eşiğini güncelle
        recommender.trend_analyzer.set_discount_threshold(min_discount)
        deals = recommender.trend_analyzer.find_deal_products(recommender.df)
        
        # Fiyat filtresi uygula
        if not deals.empty:
            deals = deals[deals['price'] <= max_price]
    
    if not deals.empty:
        insights = recommender.trend_analyzer.get_deal_insights(deals)
        
        # Özet bilgiler
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Toplam Fırsat", insights['total_deals'])
        with col2:
            st.metric("📊 Ortalama İndirim", f"%{insights['avg_discount']:.1f}")
        with col3:
            st.metric("🔥 Max İndirim", f"%{insights['max_discount']:.1f}")
        with col4:
            if 'savings_info' in insights:
                st.metric("💰 Toplam Tasarruf", f"{insights['savings_info']['total_savings']:,.0f} ₺")
        
        st.subheader(f"🏆 En İyi {min(show_count, len(deals))} Fırsat")
        
        # Seçilen sayıda fırsatı göster
        top_deals = deals.head(show_count)
        
        for i, (_, deal) in enumerate(top_deals.iterrows(), 1):
            discount = deal.get('discount_percentage', 0)
            
            # Fırsat seviyesi
            if discount >= 40:
                level = "🔥 MUHTEŞEM"
                color = "#ff4757"
            elif discount >= 25:
                level = "⭐ ÇOK İYİ"  
                color = "#ffa502"
            else:
                level = "✨ İYİ"
                color = "#2ed573"
            
            # Ürün kartı
            with st.container():
                st.markdown(f"""
                <div style="background: {color}; padding: 1.5rem; border-radius: 15px; margin: 1rem 0; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h3>{i}. {deal['name']}</h3>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin: 1rem 0;">
                        <div>
                            <p style="font-size: 1.2em; margin: 0;"><strong>{level} FIRSAT</strong></p>
                            <p style="font-size: 1.1em; margin: 0;">📉 <strong>%{discount:.1f} İNDİRİM</strong></p>
                        </div>
                        <div style="text-align: right;">
                            <p style="font-size: 1.4em; margin: 0;"><strong>💰 {deal['price']:,} ₺</strong></p>
                            <p style="font-size: 0.9em; margin: 0; opacity: 0.8;">Piyasa: {deal.get('market_price', deal['price']*1.3):,.0f} ₺</p>
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0;">
                        <div>
                            <p style="margin: 0.2rem 0;">💾 <strong>RAM:</strong> {int(deal['ram_gb'])} GB</p>
                            <p style="margin: 0.2rem 0;">💿 <strong>SSD:</strong> {int(deal['ssd_gb'])} GB</p>
                        </div>
                        <div>
                            <p style="margin: 0.2rem 0;">🎮 <strong>GPU:</strong> {deal['gpu_clean'].upper()}</p>
                            <p style="margin: 0.2rem 0;">⚡ <strong>CPU:</strong> {deal['cpu_clean'].upper()}</p>
                        </div>
                        <div>
                            <p style="margin: 0.2rem 0;">📱 <strong>Ekran:</strong> {deal['screen_size']}"</p>
                            <p style="margin: 0.2rem 0;">🏢 <strong>Marka:</strong> {deal.get('brand', 'Bilinmiyor').title()}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # URL butonu - Daha belirgin
                col_left, col_center, col_right = st.columns([1, 2, 1])
                with col_center:
                    # URL'yi kontrol et ve düzelt
                    product_url = deal['url']
                    if pd.notna(product_url) and str(product_url).strip():
                        # URL'yi temizle
                        clean_url = str(product_url).strip()
                        if not clean_url.startswith(('http://', 'https://')):
                            clean_url = f"https://{clean_url}"
                        
                        # Link butonu
                        st.markdown(f"""
                        <div style="text-align: center; margin: 1rem 0;">
                            <a href="{clean_url}" target="_blank" style="
                                background: linear-gradient(45deg, #667eea, #764ba2);
                                color: white;
                                padding: 12px 30px;
                                border-radius: 25px;
                                text-decoration: none;
                                font-weight: bold;
                                font-size: 1.1em;
                                display: inline-block;
                                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                                transition: all 0.3s ease;
                            ">
                                🔗 FIRSATI YAKALA - ÜRÜNÜ İNCELE
                            </a>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ Ürün linki bulunamadı")
                
                st.divider()
        
        # Sayfalama önerisi
        if len(deals) > show_count:
            remaining = len(deals) - show_count
            st.info(f"ℹ️ {remaining} fırsat daha var! Gösterilecek sayıyı artırarak tümünü görebilirsiniz.")
        
        # Fırsat istatistikleri
        with st.expander("📊 Detaylı Fırsat İstatistikleri"):
            discount_stats = recommender.trend_analyzer.get_discount_statistics(deals)
            
            if discount_stats:
                st.subheader("📉 İndirim Dağılımı")
                dist = discount_stats['discount_distribution']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"{min_discount}-30%", dist.get(f'{min_discount}-30%', 0))
                with col2:
                    st.metric("30-50%", dist.get('30-50%', 0))
                with col3:
                    st.metric("50%+", dist.get('50%+', 0))
                
                # En iyi fırsatlar tablosu
                if 'top_deals' in discount_stats:
                    st.subheader("🏆 En Yüksek İndirimli Ürünler")
                    top_discount_df = pd.DataFrame(discount_stats['top_deals'])
                    if not top_discount_df.empty:
                        st.dataframe(
                            top_discount_df[['name', 'price', 'market_price', 'discount_percentage']],
                            column_config={
                                'name': 'Ürün Adı',
                                'price': st.column_config.NumberColumn('Fiyat (₺)', format="%.0f"),
                                'market_price': st.column_config.NumberColumn('Piyasa Fiyatı (₺)', format="%.0f"),
                                'discount_percentage': st.column_config.NumberColumn('İndirim %', format="%.1f")
                            },
                            use_container_width=True
                        )
    
    else:
        st.info("🤷‍♂️ Belirtilen kriterlerde fırsat bulunmuyor. Filtreleri değiştirmeyi deneyin!")
        
        # Alternatif öneriler
        st.subheader("💡 Öneriler")
        st.markdown("""
        - İndirim oranını düşürmeyi deneyin
        - Maksimum fiyat limitini artırın  
        - Farklı kategorilerde arama yapın
        """)
        
        # Genel fırsat arama butonu
        if st.button("🔍 Tüm Kategorilerde Fırsat Ara", type="secondary"):
            with st.spinner("🔄 Geniş arama yapılıyor..."):
                # Daha geniş kriterlerle arama
                recommender.trend_analyzer.set_discount_threshold(10)  # %10'a düşür
                broader_deals = recommender.trend_analyzer.find_deal_products(recommender.df)
                
                if not broader_deals.empty:
                    st.success(f"✅ {len(broader_deals)} alternatif fırsat bulundu!")
                    
                    # İlk 5'ini göster
                    for i, (_, deal) in enumerate(broader_deals.head(5).iterrows(), 1):
                        st.markdown(f"**{i}. {deal['name'][:50]}...** - {deal['price']:,} ₺ (%{deal.get('discount_percentage', 0):.1f} indirim)")
                else:
                    st.warning("Geniş aramada da fırsat bulunamadı.")

def show_analytics_page(recommender):
    """Analiz sayfası"""
    st.header("📊 Pazar Analizi ve Trendler")
    
    df = recommender.df
    
    # Genel istatistikler
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Fiyat Analizi")
        
        # Fiyat dağılımı
        fig_hist = px.histogram(
            df, x='price', nbins=30,
            title="Laptop Fiyat Dağılımı",
            labels={'price': 'Fiyat (₺)', 'count': 'Adet'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Fiyat istatistikleri
        st.metric("Ortalama Fiyat", f"{df['price'].mean():,.0f} ₺")
        st.metric("Medyan Fiyat", f"{df['price'].median():,.0f} ₺")
        st.metric("En Pahalı", f"{df['price'].max():,.0f} ₺")
        st.metric("En Ucuz", f"{df['price'].min():,.0f} ₺")
    
    with col2:
        st.subheader("🏢 Marka Analizi")
        
        # Marka sayıları
        brand_counts = df['brand'].value_counts()
        fig_brand_pie = px.pie(
            values=brand_counts.values,
            names=[b.title() for b in brand_counts.index],
            title="Marka Dağılımı"
        )
        st.plotly_chart(fig_brand_pie, use_container_width=True)
        
        # Marka ortalama fiyatları
        brand_avg_price = df.groupby('brand')['price'].mean().sort_values(ascending=False)
        fig_brand_price = px.bar(
            x=[b.title() for b in brand_avg_price.index],
            y=brand_avg_price.values,
            title="Marka Ortalama Fiyatları"
        )
        st.plotly_chart(fig_brand_price, use_container_width=True)
    
    # GPU Performans Analizi
    st.subheader("🎮 GPU Performans Analizi")
    
    gpu_data = df.groupby('gpu_clean').agg({
        'price': 'mean',
        'gpu_score': 'first',
        'name': 'count'
    }).rename(columns={'name': 'count'})
    
    gpu_data = gpu_data[gpu_data['count'] >= 3].sort_values('gpu_score', ascending=False)
    
    fig_gpu = px.scatter(
        gpu_data, 
        x='gpu_score', 
        y='price',
        size='count',
        hover_name=gpu_data.index,
        title="GPU Skoru vs Ortalama Fiyat",
        labels={'gpu_score': 'GPU Performans Skoru', 'price': 'Ortalama Fiyat (₺)'}
    )
    st.plotly_chart(fig_gpu, use_container_width=True)

if __name__ == "__main__":
    main()