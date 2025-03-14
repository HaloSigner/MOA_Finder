import streamlit as st
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures

# ✅ PubMed API Base URL
PUBMED_API_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# ✅ Define save directory
SAVE_DIR = "./pubmed_results/"
os.makedirs(SAVE_DIR, exist_ok=True)  # ✅ Ensure directory exists

# ✅ Database file for MOA storage
MOA_DATABASE_FILE = os.path.join(SAVE_DIR, "moa_database.csv")

# ✅ NCBI API Key (Set your API key here)
NCBI_API_KEY = "c8b00d6f453fc91be771fc407cf334959c08"  # 🔴 Replace with your actual API key

# ✅ MOA-related keyword filtering list
MOA_KEYWORDS = ["Inhibition", "Activation", "Antagonist", "Agonist", "Receptor", "Kinase", "Signal Transduction"]

# ✅ Load existing MOA database
def load_moa_database():
    if os.path.exists(MOA_DATABASE_FILE):
        return pd.read_csv(MOA_DATABASE_FILE)
    return pd.DataFrame(columns=["Drug", "MOA_1", "MOA_2"])

# ✅ Save MOA database
def save_moa_database(df):
    df.to_csv(MOA_DATABASE_FILE, index=False, encoding="utf-8")

# ✅ Retry API requests with backoff
def retry_with_backoff(url, retries=3, sleep_time=1.5):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response
            time.sleep(sleep_time)
        except requests.exceptions.RequestException:
            time.sleep(sleep_time)
    return None

# ✅ Search PubMed for PMIDs
def search_pubmed_parallel(drugs, max_results=500):
    def search_for_drug(drug):
        query = f"{drug} mechanism of action"
        pmid_list = []
        batch_size = 500

        for start in range(0, max_results, batch_size):
            search_url = f"{PUBMED_API_URL}esearch.fcgi?db=pubmed&term={query}&retstart={start}&retmax={batch_size}&retmode=xml&api_key={NCBI_API_KEY}"
            response = retry_with_backoff(search_url)
            if not response:
                continue

            root = ET.fromstring(response.text)
            batch_pmids = [id_elem.text for id_elem in root.findall(".//Id")]
            if not batch_pmids:
                break
            pmid_list.extend(batch_pmids)
            time.sleep(0.5)
        return drug, pmid_list

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(search_for_drug, drugs))

    return {drug: pmids for drug, pmids in results}

# ✅ Retrieve MeSH terms for PMIDs
def fetch_mesh_terms_parallel(pmid_dict, batch_size=100):
    def fetch_batch(pmids, batch_pmids):
        details_url = f"{PUBMED_API_URL}efetch.fcgi?db=pubmed&id={','.join(batch_pmids)}&retmode=xml&api_key={NCBI_API_KEY}"
        response = retry_with_backoff(details_url)
        if response:
            root = ET.fromstring(response.text)
            return [mesh.text for mesh in root.findall(".//PubmedArticle//MeshHeading/DescriptorName")]
        return []

    mesh_terms_dict = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for drug, pmids in pmid_dict.items():
            mesh_terms = []
            batches = [pmids[i:i + batch_size] for i in range(0, len(pmids), batch_size)]
            future_to_batch = {executor.submit(fetch_batch, pmids, batch): batch for batch in batches}

            for future in concurrent.futures.as_completed(future_to_batch):
                mesh_terms.extend(future.result())

            mesh_terms_dict[drug] = mesh_terms
            time.sleep(0.5)

    return mesh_terms_dict

# ✅ Identify top MOA-related MeSH terms
def get_most_common_moa(mesh_terms):
    mesh_counts = Counter(mesh_terms)
    sorted_mesh = sorted(mesh_counts.items(), key=lambda x: x[1], reverse=True)
    moa_terms = [term for term, freq in sorted_mesh if any(keyword in term for keyword in MOA_KEYWORDS)]
    return moa_terms[:2] if moa_terms else ["No MOA Found", "No MOA Found"]

# ✅ Fetch PubChem structure image URL
def fetch_pubchem_structure(drug):
    """Fetch PubChem compound structure image (PNG)."""
    return f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug}/PNG"

def display_pubchem_structure(drug):
    """Display PubChem structure image in Streamlit."""
    image_url = fetch_pubchem_structure(drug)
    st.image(image_url, caption=f"Structure of {drug}", use_container_width=True)

def fetch_pubchem_info(drug):
    """Fetch detailed information from PubChem based on the drug name."""
    pubchem_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug}/JSON"
    
    try:
        response = requests.get(pubchem_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "PC_Compounds" in data and data["PC_Compounds"]:
                compound_data = data["PC_Compounds"][0]
                
                # Extract key information
                compound_info = {
                    "CID": compound_data.get("id", {}).get("id", "N/A"),
                    "IUPAC": compound_data.get("props", [{}])[0].get("value", {}).get("sval", "N/A"),
                    "MolecularWeight": compound_data.get("props", [{}])[1].get("value", {}).get("fval", "N/A"),
                    "Synonyms": compound_data.get("synonyms", ["No synonyms found"]),
                    "Image": f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{compound_data.get('id', {}).get('id', 'N/A')}/PNG"
                }

                # Handle missing values
                for key in compound_info:
                    if compound_info[key] is None or (isinstance(compound_info[key], float) and pd.isna(compound_info[key])):
                        compound_info[key] = 'N/A'

                return compound_info
        return None
    except requests.exceptions.RequestException:
        return None

# ✅ Bar chart for MOA distribution
def plot_moa_distribution_by_category(moa_db):
    category_counts = {key: 0 for key in MOA_KEYWORDS}

    for _, row in moa_db.iterrows():
        for keyword in MOA_KEYWORDS:
            if keyword in row['MOA_1']:
                category_counts[keyword] += 1
            if keyword in row['MOA_2']:
                category_counts[keyword] += 1

    st.write("MOA Category Distribution:")
    st.bar_chart(category_counts)

# ✅ Streamlit UI
st.set_page_config(page_title="MOA Search System", layout="wide")
st.title("🧪 MOA Search System")
st.sidebar.title("🔬 MOA Search System")
st.sidebar.markdown("Enter drug names below to retrieve **MOA and PubChem Structure**.")


col1 ,col2 = st.columns([2,2])

with col1: 
    drug_input = st.text_input("🔍 Enter drug names (comma-separated)")

with col2:
    max_articles = st.number_input("🔢 Max articles per drug", min_value=10, max_value=1000, value=500)

# ✅ Search button
if st.button("Search MOA"):
    if not drug_input:
        st.warning("⚠️ Please enter at least one drug name.")
    else:
        drugs = [drug.strip() for drug in drug_input.split(",") if drug.strip()]

        moa_db = load_moa_database()
        new_results = []

        pmid_dict = search_pubmed_parallel(drugs, max_results=max_articles)
        mesh_terms_dict = fetch_mesh_terms_parallel(pmid_dict, batch_size=100)

        result_data = []

    for drug in drugs:
        progress_bar = st.progress(0)
        st.write(f"🔬 **Processing {drug}...**")

        # Get MeSH terms for the drug
        mesh_terms = mesh_terms_dict.get(drug, [])
    moa_terms = get_most_common_moa(mesh_terms)

    # Fetch PubChem information
    pubchem_info = fetch_pubchem_info(drug)

    with st.expander(f"🔹 **{drug} - MOA & PubChem Information**", expanded=False):
        # ✅ Structure Image
        display_pubchem_structure(drug)

        # ✅ PubChem Information
        if pubchem_info:
            st.write(f"**PubChem Information for {drug}:**")
            st.write(f"- **CID**: {pubchem_info['CID']}")
            st.write(f"- **IUPAC Name**: {pubchem_info['IUPAC']}")
            st.write(f"- **Molecular Weight**: {pubchem_info['MolecularWeight']}")
            st.write(f"- **Synonyms**: {', '.join(pubchem_info['Synonyms']) if pubchem_info['Synonyms'] else 'No synonyms found'}")
        else:
            st.warning(f"❌ PubChem information not found for {drug}")

    # ✅ MOA 
    st.write(f"- **MOA 1**: {moa_terms[0]}")
    st.write(f"- **MOA 2**: {moa_terms[1]}")

    progress_bar.progress(100)

    if result_data:
        new_df = pd.DataFrame(result_data)
        moa_db = pd.concat([moa_db, new_df], ignore_index=True)
        save_moa_database(moa_db)

    st.success("✅ MOA search complete!")
    st.write("### Results")
    st.dataframe(pd.DataFrame(result_data))
    plot_moa_distribution_by_category(moa_db)