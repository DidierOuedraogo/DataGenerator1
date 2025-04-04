import streamlit as st
import numpy as np
import pandas as pd
import base64
from io import BytesIO
import datetime

st.set_page_config(page_title="Générateur de données minières synthétiques", layout="wide")

st.title("Générateur de données minières synthétiques")
st.caption("Développé par Didier Ouedraogo, P.Geo.")

# Sidebar pour les paramètres
st.sidebar.header("Paramètres")

# Section pour choisir le type de données
data_type = st.sidebar.radio(
    "Type de données à générer",
    ["Composites", "Modèle de bloc", "Données QAQC"]
)

# Liste des métaux disponibles
available_metals = ["Or", "Cuivre", "Zinc", "Manganèse", "Fer"]
selected_metals = st.sidebar.multiselect(
    "Sélectionnez les métaux",
    available_metals,
    default=["Or"]
)

# Fonction pour générer des données de composites
def generate_composite_data(num_samples, composite_size, mean_values, std_values, selected_metals):
    # Créer un DataFrame avec coordonnées X, Y, Z aléatoires
    data = {
        'Composite_ID': range(1, num_samples + 1),
        'X': np.random.uniform(0, 1000, num_samples),
        'Y': np.random.uniform(0, 1000, num_samples),
        'Z': np.random.uniform(-500, 0, num_samples),
        'Longueur': np.ones(num_samples) * composite_size
    }
    
    # Ajouter les teneurs pour chaque métal sélectionné
    for i, metal in enumerate(selected_metals):
        # Générer des données selon une distribution log-normale pour plus de réalisme
        data[f'Teneur_{metal}'] = np.random.lognormal(
            mean=np.log(mean_values[i]) - 0.5 * np.log(1 + (std_values[i]/mean_values[i])**2),
            sigma=np.sqrt(np.log(1 + (std_values[i]/mean_values[i])**2)),
            size=num_samples
        )
    
    return pd.DataFrame(data)

# Fonction pour générer un modèle de bloc
def generate_block_model(nx, ny, nz, block_size_x, block_size_y, block_size_z, means, stds, selected_metals):
    num_blocks = nx * ny * nz
    
    # Créer les indices de grille
    x_indices = np.repeat(np.arange(nx), ny * nz)
    y_indices = np.tile(np.repeat(np.arange(ny), nz), nx)
    z_indices = np.tile(np.arange(nz), nx * ny)
    
    # Calculer les coordonnées centrales
    x = x_indices * block_size_x + block_size_x / 2
    y = y_indices * block_size_y + block_size_y / 2
    z = z_indices * block_size_z + block_size_z / 2
    
    # Créer le DataFrame du modèle de bloc
    data = {
        'Block_ID': np.arange(1, num_blocks + 1),
        'X': x,
        'Y': y,
        'Z': z,
        'X_size': np.ones(num_blocks) * block_size_x,
        'Y_size': np.ones(num_blocks) * block_size_y,
        'Z_size': np.ones(num_blocks) * block_size_z
    }
    
    # Simuler une continuité spatiale simple pour les teneurs (corrélation spatiale)
    for i, metal in enumerate(selected_metals):
        # Créer un champ de base avec une certaine corrélation spatiale
        # Cette approche simplifie la géostatistique réelle mais donne un aspect plus réaliste
        base_field = np.zeros((nx, ny, nz))
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for dz in range(-2, 3):
                    weight = 1.0 / (1.0 + np.sqrt(dx**2 + dy**2 + dz**2))
                    random_field = np.random.normal(0, 1, (nx, ny, nz))
                    # Ajouter avec décalage et gestion des bords
                    x_start, x_end = max(0, dx), min(nx, nx + dx)
                    y_start, y_end = max(0, dy), min(ny, ny + dy)
                    z_start, z_end = max(0, dz), min(nz, nz + dz)
                    
                    x_target_start, x_target_end = max(0, -dx), min(nx, nx - dx)
                    y_target_start, y_target_end = max(0, -dy), min(ny, ny - dy)
                    z_target_start, z_target_end = max(0, -dz), min(nz, nz - dz)
                    
                    base_field[x_target_start:x_target_end, y_target_start:y_target_end, z_target_start:z_target_end] += (
                        weight * random_field[x_start:x_end, y_start:y_end, z_start:z_end]
                    )
        
        # Normaliser et transformer le champ pour obtenir la distribution souhaitée
        base_field = (base_field - np.mean(base_field)) / np.std(base_field)
        base_field_flattened = base_field.flatten()
        
        # Convertir en distribution log-normale avec les paramètres souhaités
        log_mean = np.log(means[i]) - 0.5 * np.log(1 + (stds[i]/means[i])**2)
        log_std = np.sqrt(np.log(1 + (stds[i]/means[i])**2))
        values = np.exp(log_mean + log_std * base_field_flattened)
        
        data[f'Teneur_{metal}'] = values
    
    return pd.DataFrame(data)

# Fonction pour générer des données QAQC
def generate_qaqc_data(num_samples, crm_percent, duplicate_percent, blank_percent, crm_values, crm_std_values, selected_metals):
    # Calculer le nombre d'échantillons de chaque type
    num_crm = int(num_samples * crm_percent / 100)
    num_duplicates = int(num_samples * duplicate_percent / 100)
    num_blanks = int(num_samples * blank_percent / 100)
    num_regular = num_samples - num_crm - num_duplicates - num_blanks
    
    # Vérifier que nous avons au moins un échantillon régulier pour créer des duplicatas
    if num_regular <= 0:
        st.error("Les pourcentages sont trop élevés. Impossible de générer des échantillons réguliers.")
        return None
    
    # Générer des dates pour simulation d'une campagne de forage
    start_date = datetime.datetime.now() - datetime.timedelta(days=180)
    dates = [start_date + datetime.timedelta(days=x) for x in np.sort(np.random.randint(0, 180, num_samples))]
    date_strings = [d.strftime("%Y-%m-%d") for d in dates]
    
    # Générer des échantillons réguliers
    regular_data = {
        'Sample_ID': [f"REG-{i+1:06d}" for i in range(num_regular)],
        'Batch_ID': np.random.randint(1000, 2000, num_regular),
        'Date': date_strings[:num_regular],
        'Type': ['Regular'] * num_regular,
        'Trou_ID': [f"DDH-{i+1:03d}" for i in np.random.randint(1, 51, num_regular)],
        'De': np.random.uniform(0, 500, num_regular),
    }
    
    # Ajouter la colonne "À" (profondeur de fin) basée sur "De" + une longueur aléatoire
    sample_lengths = np.random.uniform(0.5, 2.5, num_regular)
    regular_data['A'] = regular_data['De'] + sample_lengths
    
    # Générer des teneurs pour les échantillons réguliers
    for i, metal in enumerate(selected_metals):
        # Utiliser une distribution log-normale pour simuler des teneurs réalistes
        log_mean = np.log(crm_values[i]) - 0.5 * np.log(1 + (crm_std_values[i]/crm_values[i])**2)
        log_std = np.sqrt(np.log(1 + (crm_std_values[i]/crm_values[i])**2))
        
        # Simuler une variabilité spatiale en utilisant une tendance en fonction de la profondeur
        base_values = np.exp(log_mean + log_std * np.random.normal(0, 1, num_regular))
        depth_effect = 1 + 0.3 * np.sin(regular_data['De'] / 50)  # Une légère tendance cyclique avec la profondeur
        regular_data[f'Teneur_{metal}'] = base_values * depth_effect
    
    # Créer le DataFrame pour les échantillons réguliers
    regular_df = pd.DataFrame(regular_data)
    
    # Générer des CRM (matériaux de référence certifiés)
    if num_crm > 0:
        crm_types = [f"CRM-{chr(65+i)}" for i in range(min(3, len(selected_metals)))]  # Jusqu'à 3 types de CRM
        crm_data = {
            'Sample_ID': [f"CRM-{i+1:06d}" for i in range(num_crm)],
            'Batch_ID': np.random.randint(1000, 2000, num_crm),
            'Date': date_strings[num_regular:num_regular+num_crm],
            'Type': ['CRM'] * num_crm,
            'CRM_Type': np.random.choice(crm_types, num_crm),
            'Trou_ID': [f"QC" for _ in range(num_crm)],
            'De': [0] * num_crm,
            'A': [0] * num_crm
        }
        
        # Ajouter les teneurs certifiées pour chaque CRM avec une légère variation
        for i, metal in enumerate(selected_metals):
            crm_values_per_type = {
                crm_type: crm_values[i] * (0.8 + j*0.2)  # Différentes valeurs pour différents types de CRM
                for j, crm_type in enumerate(crm_types)
            }
            
            crm_data[f'Teneur_{metal}'] = [
                np.random.normal(
                    crm_values_per_type[crm_type], 
                    crm_std_values[i] * 0.5  # Précision plus élevée pour les CRM
                ) 
                for crm_type in crm_data['CRM_Type']
            ]
        
        crm_df = pd.DataFrame(crm_data)
    else:
        crm_df = pd.DataFrame()
    
    # Générer des duplicatas
    if num_duplicates > 0 and num_regular > 0:
        # Sélectionner aléatoirement des échantillons à dupliquer
        original_indices = np.random.choice(range(num_regular), min(num_duplicates, num_regular), replace=False)
        original_samples = regular_df.iloc[original_indices].copy()
        
        duplicates_data = {
            'Sample_ID': [f"DUP-{i+1:06d}" for i in range(len(original_indices))],
            'Batch_ID': np.random.randint(1000, 2000, len(original_indices)),
            'Date': date_strings[num_regular+num_crm:num_regular+num_crm+len(original_indices)],
            'Type': ['Duplicate'] * len(original_indices),
            'Original_Sample': original_samples['Sample_ID'].values,
            'Trou_ID': original_samples['Trou_ID'].values,
            'De': original_samples['De'].values,
            'A': original_samples['A'].values
        }
        
        # Ajouter les teneurs avec une légère variation par rapport aux originaux
        for i, metal in enumerate(selected_metals):
            duplicates_data[f'Teneur_{metal}'] = original_samples[f'Teneur_{metal}'].values * np.random.normal(1, 0.05, len(original_indices))
        
        duplicates_df = pd.DataFrame(duplicates_data)
    else:
        duplicates_df = pd.DataFrame()
    
    # Générer des blancs
    if num_blanks > 0:
        blank_data = {
            'Sample_ID': [f"BLK-{i+1:06d}" for i in range(num_blanks)],
            'Batch_ID': np.random.randint(1000, 2000, num_blanks),
            'Date': date_strings[num_regular+num_crm+num_duplicates:],
            'Type': ['Blank'] * num_blanks,
            'Trou_ID': [f"QC" for _ in range(num_blanks)],
            'De': [0] * num_blanks,
            'A': [0] * num_blanks
        }
        
        # Ajouter des teneurs très basses pour les blancs (avec occasionnellement une contamination)
        for metal in selected_metals:
            # La plupart des blancs auront des valeurs presque nulles, mais quelques-uns auront une légère contamination
            contamination = np.random.choice([0, 1], num_blanks, p=[0.95, 0.05])
            blank_data[f'Teneur_{metal}'] = np.random.lognormal(-4, 0.5, num_blanks) * (1 + contamination * 5)
        
        blanks_df = pd.DataFrame(blank_data)
    else:
        blanks_df = pd.DataFrame()
    
    # Combiner tous les types d'échantillons
    dfs_to_combine = [df for df in [regular_df, crm_df, duplicates_df, blanks_df] if not df.empty]
    qaqc_data = pd.concat(dfs_to_combine, ignore_index=True)
    
    return qaqc_data

# Fonction pour télécharger le DataFrame en CSV
def get_csv_download_link(df, filename="data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Télécharger le fichier CSV</a>'
    return href

# Interface utilisateur en fonction du type de données choisi
if data_type == "Composites":
    st.header("Génération de données de composites")
    
    num_composites = st.slider("Nombre de composites", 10, 10000, 1000)
    composite_size = st.slider("Taille des composites (m)", 0.5, 10.0, 2.0)
    
    # Paramètres pour chaque métal sélectionné
    st.subheader("Paramètres des teneurs")
    mean_values = []
    std_values = []
    
    metal_units = {
        "Or": "g/t",
        "Cuivre": "%",
        "Zinc": "%",
        "Manganèse": "%",
        "Fer": "%"
    }
    
    default_values = {
        "Or": (1.5, 0.8),
        "Cuivre": (0.5, 0.2),
        "Zinc": (2.0, 1.0),
        "Manganèse": (1.2, 0.5),
        "Fer": (30.0, 10.0)
    }
    
    # Création d'une mise en page en colonnes pour les paramètres des métaux
    cols = st.columns(len(selected_metals) if selected_metals else 1)
    
    for i, metal in enumerate(selected_metals):
        with cols[i]:
            st.write(f"**{metal}** ({metal_units[metal]})")
            mean = st.number_input(f"Teneur moyenne - {metal}", 
                                 min_value=0.001, 
                                 max_value=100.0, 
                                 value=float(default_values[metal][0]),
                                 step=0.1,
                                 key=f"mean_{metal}")
            std = st.number_input(f"Écart-type - {metal}", 
                                min_value=0.001, 
                                max_value=50.0, 
                                value=float(default_values[metal][1]),
                                step=0.1,
                                key=f"std_{metal}")
            mean_values.append(mean)
            std_values.append(std)
    
    if st.button("Générer les données de composites"):
        if selected_metals:
            # Générer les données
            composite_data = generate_composite_data(num_composites, composite_size, mean_values, std_values, selected_metals)
            
            # Afficher les statistiques des données
            st.subheader("Statistiques des données générées")
            st.dataframe(composite_data.describe())
            
            # Afficher un échantillon des données
            st.subheader("Aperçu des données")
            st.dataframe(composite_data.head(10))
            
            # Lien de téléchargement
            st.markdown(get_csv_download_link(composite_data, "composites_data.csv"), unsafe_allow_html=True)
        else:
            st.error("Veuillez sélectionner au moins un métal.")

elif data_type == "Modèle de bloc":
    st.header("Génération de modèle de bloc")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dimensions du modèle")
        nx = st.slider("Nombre de blocs en X", 5, 100, 20)
        ny = st.slider("Nombre de blocs en Y", 5, 100, 20)
        nz = st.slider("Nombre de blocs en Z", 5, 50, 10)
    
    with col2:
        st.subheader("Taille des blocs (m)")
        block_size_x = st.slider("Taille du bloc en X", 1.0, 50.0, 10.0)
        block_size_y = st.slider("Taille du bloc en Y", 1.0, 50.0, 10.0)
        block_size_z = st.slider("Taille du bloc en Z", 1.0, 25.0, 5.0)
    
    # Calculer et afficher le nombre total de blocs
    total_blocks = nx * ny * nz
    st.info(f"Nombre total de blocs: {total_blocks:,}")
    
    # Paramètres pour chaque métal sélectionné
    st.subheader("Paramètres des teneurs")
    mean_values = []
    std_values = []
    
    metal_units = {
        "Or": "g/t",
        "Cuivre": "%",
        "Zinc": "%",
        "Manganèse": "%",
        "Fer": "%"
    }
    
    default_values = {
        "Or": (1.0, 0.5),
        "Cuivre": (0.4, 0.15),
        "Zinc": (1.5, 0.7),
        "Manganèse": (1.0, 0.4),
        "Fer": (25.0, 8.0)
    }
    
    # Création d'une mise en page en colonnes pour les paramètres des métaux
    cols = st.columns(len(selected_metals) if selected_metals else 1)
    
    for i, metal in enumerate(selected_metals):
        with cols[i]:
            st.write(f"**{metal}** ({metal_units[metal]})")
            mean = st.number_input(f"Teneur moyenne - {metal}", 
                                 min_value=0.001, 
                                 max_value=100.0, 
                                 value=float(default_values[metal][0]),
                                 step=0.1,
                                 key=f"bm_mean_{metal}")
            std = st.number_input(f"Écart-type - {metal}", 
                                min_value=0.001, 
                                max_value=50.0, 
                                value=float(default_values[metal][1]),
                                step=0.1,
                                key=f"bm_std_{metal}")
            mean_values.append(mean)
            std_values.append(std)
    
    if st.button("Générer le modèle de bloc"):
        if selected_metals:
            if total_blocks > 500000:
                if not st.warning("Le modèle va générer un grand nombre de blocs, ce qui peut prendre du temps. Continuer?"):
                    st.stop()
            
            # Afficher un indicateur de progression
            progress_bar = st.progress(0)
            st.write("Génération du modèle de bloc en cours...")
            
            # Générer les données
            block_model = generate_block_model(nx, ny, nz, block_size_x, block_size_y, block_size_z, 
                                               mean_values, std_values, selected_metals)
            
            progress_bar.progress(100)
            
            # Afficher les statistiques des données
            st.subheader("Statistiques des données générées")
            st.dataframe(block_model.describe())
            
            # Afficher un échantillon des données
            st.subheader("Aperçu des données")
            st.dataframe(block_model.head(10))
            
            # Lien de téléchargement
            st.markdown(get_csv_download_link(block_model, "block_model_data.csv"), unsafe_allow_html=True)
        else:
            st.error("Veuillez sélectionner au moins un métal.")

elif data_type == "Données QAQC":
    st.header("Génération de données QAQC")
    
    num_samples = st.slider("Nombre total d'échantillons", 50, 5000, 500)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        crm_percent = st.slider("Pourcentage de CRM (%)", 0, 30, 5)
    
    with col2:
        duplicate_percent = st.slider("Pourcentage de duplicatas (%)", 0, 30, 5)
    
    with col3:
        blank_percent = st.slider("Pourcentage de blancs (%)", 0, 30, 3)
    
    # Vérifier que le total ne dépasse pas 100%
    total_percent = crm_percent + duplicate_percent + blank_percent
    if total_percent > 50:
        st.warning(f"Le total des pourcentages de QAQC est de {total_percent}%. Il est recommandé de ne pas dépasser 50%.")
    
    # Paramètres pour les CRM (pour chaque métal)
    st.subheader("Paramètres des CRM")
    crm_values = []
    crm_std_values = []
    
    metal_units = {
        "Or": "g/t",
        "Cuivre": "%",
        "Zinc": "%",
        "Manganèse": "%",
        "Fer": "%"
    }
    
    default_values = {
        "Or": (2.5, 0.1),
        "Cuivre": (0.7, 0.03),
        "Zinc": (3.0, 0.15),
        "Manganèse": (1.5, 0.07),
        "Fer": (40.0, 2.0)
    }
    
    # Création d'une mise en page en colonnes pour les paramètres des métaux
    cols = st.columns(len(selected_metals) if selected_metals else 1)
    
    for i, metal in enumerate(selected_metals):
        with cols[i]:
            st.write(f"**{metal}** ({metal_units[metal]})")
            value = st.number_input(f"Valeur certifiée - {metal}", 
                                   min_value=0.001, 
                                   max_value=100.0, 
                                   value=float(default_values[metal][0]),
                                   step=0.1,
                                   key=f"crm_val_{metal}")
            std = st.number_input(f"Écart-type toléré - {metal}", 
                                min_value=0.001, 
                                max_value=10.0, 
                                value=float(default_values[metal][1]),
                                step=0.01,
                                key=f"crm_std_{metal}")
            crm_values.append(value)
            crm_std_values.append(std)
    
    if st.button("Générer les données QAQC"):
        if selected_metals:
            # Générer les données
            qaqc_data = generate_qaqc_data(num_samples, crm_percent, duplicate_percent, blank_percent, 
                                           crm_values, crm_std_values, selected_metals)
            
            # Afficher les statistiques des données
            st.subheader("Statistiques des données générées")
            st.dataframe(qaqc_data.describe())
            
            # Afficher la distribution des types d'échantillons
            st.subheader("Distribution des types d'échantillons")
            type_counts = qaqc_data['Type'].value_counts()
            st.write(type_counts)
            
            # Afficher un échantillon des données par type
            st.subheader("Aperçu des échantillons réguliers")
            st.dataframe(qaqc_data[qaqc_data['Type'] == 'Regular'].head(5))
            
            if 'CRM' in qaqc_data['Type'].values:
                st.subheader("Aperçu des CRM")
                st.dataframe(qaqc_data[qaqc_data['Type'] == 'CRM'].head(5))
            
            if 'Duplicate' in qaqc_data['Type'].values:
                st.subheader("Aperçu des duplicatas")
                st.dataframe(qaqc_data[qaqc_data['Type'] == 'Duplicate'].head(5))
            
            if 'Blank' in qaqc_data['Type'].values:
                st.subheader("Aperçu des blancs")
                st.dataframe(qaqc_data[qaqc_data['Type'] == 'Blank'].head(5))
            
            # Lien de téléchargement
            st.markdown(get_csv_download_link(qaqc_data, "qaqc_data.csv"), unsafe_allow_html=True)
        else:
            st.error("Veuillez sélectionner au moins un métal.")

# Footer
st.markdown("---")
st.markdown("Générateur de données minières synthétiques | Développé par Didier Ouedraogo, P.Geo.")