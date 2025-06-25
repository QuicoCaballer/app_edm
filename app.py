import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point
from shapely.errors import GEOSException
from shapely.ops import unary_union
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors
import altair as alt

# -- Utility Functions --

def load_data(path="contenedores.csv"):
    df = pd.read_csv(path, encoding="ISO-8859-1", sep=";", on_bad_lines='skip')
    df = df.dropna(subset=["geo_point_2d"])
    df[['lat', 'lon']] = df["geo_point_2d"].str.split(",", expand=True).astype(float)
    return df


def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)
        new_regions.append(new_region)
    return new_regions, np.array(new_vertices)

# -- Safe intersection helper --
def safe_intersection(poly, bbox):
    try:
        return poly.intersection(bbox)
    except GEOSException:
        # attempt to repair
        poly2 = poly.buffer(0)
        if not poly2.is_valid:
            return None
        try:
            return poly2.intersection(bbox)
        except GEOSException:
            return None

# -- Main App --

def main():
    st.set_page_config(page_title="Valencia Limpia", layout="wide")
    st.sidebar.title("Valencia Limpia")
    page = st.sidebar.radio("Navegar a", ["Visor", "Análisis", "Optimización"])
    data = load_data()
    if page == "Visor":
        visor_page(data)
    elif page == "Análisis":
        analysis_page(data)
    else:
        optimization_page(data)

# -- Page: Visor --
def visor_page(df):
    st.header("Visor de Contenedores")
    tipo = st.selectbox("Tipo de residuo", sorted(df["Tipus Contenidor / Tipo Contenedor"].dropna().unique()))
    subset = df[df["Tipus Contenidor / Tipo Contenedor"] == tipo]
    st.markdown(f"**Contenedores encontrados:** {len(subset):,}")
    m = folium.Map(location=[39.4699, -0.3763], zoom_start=12)
    for _, row in subset.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=4,
            popup=row.get('Ubicació / Ubicación', ''),
            color='green', fill=True
        ).add_to(m)
    from folium.plugins import HeatMap
    heat = HeatMap(subset[['lat', 'lon']].values.tolist(), radius=15)
    folium.FeatureGroup(name='Heatmap').add_child(heat).add_to(m)
    points = subset[['lon','lat']].values
    if len(points) >= 3:
        vor = Voronoi(points)
        regions, verts = voronoi_finite_polygons_2d(vor)
        bbox = Polygon([[-0.6, 39.3], [-0.6, 39.7], [-0.1, 39.7], [-0.1, 39.3]])
        fg = folium.FeatureGroup(name='Voronoi')
        for region in regions:
            poly = Polygon(verts[region])
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_valid:
                continue
            clipped = safe_intersection(poly, bbox)
            if clipped is None or clipped.is_empty:
                continue
            folium.GeoJson(
                data=clipped.__geo_interface__,
                style_function=lambda x: {'fillColor':'blue','color':'blue','fillOpacity':0.1}
            ).add_to(fg)
        fg.add_to(m)
    folium.LayerControl().add_to(m)
    st_folium(m, width=1000, height=600)

# -- Page: Análisis --
def analysis_page(df):
    st.header("Análisis Estadístico")
    st.subheader("Conteo por Tipo y Distrito")
    type_counts = df['Tipus Contenidor / Tipo Contenedor'].value_counts().reset_index()
    type_counts.columns = ['Tipo', 'Cantidad']
    st.bar_chart(type_counts.set_index('Tipo'))
    if 'Distrito' in df.columns:
        dist_counts = df['Distrito'].value_counts().reset_index()
        dist_counts.columns = ['Distrito', 'Cantidad']
        st.bar_chart(dist_counts.set_index('Distrito'))
    st.subheader("Distribución de Distancias entre Vecinos Cercanos")
    coords = df[['lat','lon']].values
    nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
    dists, _ = nbrs.kneighbors(coords)
    df['nn_dist'] = dists[:,1]
    if 'Barrio' in df.columns:
        top5 = df['Barrio'].value_counts().nlargest(5).index.tolist()
        df5 = df[df['Barrio'].isin(top5)]
        box = alt.Chart(df5).mark_boxplot().encode(x='Barrio:N', y='nn_dist:Q')
        st.altair_chart(box, use_container_width=True)
        st.markdown("*Solo se muestran los 5 barrios con más contenedores.*")

# -- Page: Optimización --
def optimization_page(df):
    st.header("Optimización y Planificación")
    mode = st.selectbox("Seleccione función", ["Rutas Vecino Cercano", "Propuesta Nuevo Sitio"])
    if mode == "Rutas Vecino Cercano":
        st.subheader("Planificador de Ruta de Recolección")
        lat0 = st.number_input("Latitud inicio", value=39.4699)
        lon0 = st.number_input("Longitud inicio", value=-0.3763)
        tipo = st.selectbox("Tipo de residuo", sorted(df["Tipus Contenidor / Tipo Contenedor"].dropna().unique()))
        n = st.slider("Número de contenedores", 2, 20, 5)
        subset = df[df["Tipus Contenidor / Tipo Contenedor"] == tipo]
        pts = subset[['lat','lon']].values
        route = [(lat0, lon0)]
        remaining = list(pts)
        current = np.array([lat0, lon0])
        for _ in range(min(n, len(remaining))):
            dists = np.linalg.norm(remaining - current, axis=1)
            idx = dists.argmin()
            next_pt = remaining.pop(idx)
            route.append(tuple(next_pt))
            current = next_pt
        m = folium.Map(location=[lat0, lon0], zoom_start=13)
        folium.PolyLine(route, color='red').add_to(m)
        for i, pt in enumerate(route[1:], start=1):
            folium.Marker(pt, tooltip=f"{i}").add_to(m)
        st_folium(m, width=1000, height=600)
    else:
        st.subheader("Sugerencia de Nuevos Sitios")
        tipo = st.selectbox("Tipo de residuo", sorted(df["Tipus Contenidor / Tipo Contenedor"].dropna().unique()))
        k = st.slider("Número de nuevos sitios (K)", 1, 10, 3)
        subset = df[df["Tipus Contenidor / Tipo Contenedor"] == tipo]
        xy = subset[['lon','lat']].values.T
        kde = gaussian_kde(xy)
        lon_min, lon_max = xy[0].min(), xy[0].max()
        lat_min, lat_max = xy[1].min(), xy[1].max()
        xx, yy = np.meshgrid(np.linspace(lon_min, lon_max, 100), np.linspace(lat_min, lat_max, 100))
        grid_coords = np.vstack([xx.ravel(), yy.ravel()])
        dens = kde(grid_coords)
        thresh = np.quantile(dens, 0.2)
        low_pts = grid_coords[:, dens <= thresh].T
        kmeans = KMeans(n_clusters=k, random_state=0).fit(low_pts)
        centers = kmeans.cluster_centers_[:, ::-1]
        sil = silhouette_score(low_pts, kmeans.labels_)
        m = folium.Map(location=[39.4699, -0.3763], zoom_start=12)
        for _, row in subset.iterrows():
            folium.CircleMarker([row['lat'], row['lon']], radius=3, fill=True).add_to(m)
        for c in centers:
            folium.Marker(c, icon=folium.Icon(color='purple', icon='star')).add_to(m)
        st.markdown(f"**Silhouette score:** {sil:.2f}")
        st_folium(m, width=800, height=500)

if __name__ == '__main__':
    main()