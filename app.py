import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
import leafmap.foliumap as leafmap
import leafmap
from numerize.numerize import numerize

#Dashboard 
st.set_page_config(
    page_title="R2R DATA EXPLORER",
    page_icon="🌱",
    layout="wide")

#data


df = pd.read_csv('BD_GI, Pledge, RA_CSV_08_nov.csv',sep=';', header=None, prefix="q").iloc[2:]
df.set_index("q0", inplace = True)
df = df.dropna(how = 'all')


#Changing format to numeric and float variables
df['q65'] = pd.to_numeric(df['q65']) #Individuals
df['q16'] = pd.to_numeric(df['q16']) #Members
df['q959'] = pd.to_numeric(df['q959']) #Countries
df['q504'] = pd.to_numeric(df['q504']) #COMPANIES
df['q2296'] = pd.to_numeric(df['q2296']) #Natural SYSTEM
df['q1825'] = pd.to_numeric(df['q1825']) #Cities
df['q1392'] = pd.to_numeric(df['q1392']) #REGIONS

df['q2712'] = [float(str(i).replace(",", ".")) for i in df['q2712']]
df['q2741'] = [float(str(i).replace(",", ".")) for i in df['q2741']]
df['q2748'] = [float(str(i).replace(",", ".")) for i in df['q2748']]
df['q2755'] = [float(str(i).replace(",", ".")) for i in df['q2755']]
df['q2765'] = [float(str(i).replace(",", ".")) for i in df['q2765']]
df['q2775'] = [float(str(i).replace(",", ".")) for i in df['q2775']]
df['q2782'] = [float(str(i).replace(",", ".")) for i in df['q2782']]

df['q2713'] = [float(str(i).replace(",", ".")) for i in df['q2713']]
df['q2716'] = [float(str(i).replace(",", ".")) for i in df['q2716']]
df['q2742'] = [float(str(i).replace(",", ".")) for i in df['q2742']]
df['q2745'] = [float(str(i).replace(",", ".")) for i in df['q2745']]
df['q2749'] = [float(str(i).replace(",", ".")) for i in df['q2749']]
df['q2752'] = [float(str(i).replace(",", ".")) for i in df['q2752']]
df['q2756'] = [float(str(i).replace(",", ".")) for i in df['q2756']]
df['q2759'] = [float(str(i).replace(",", ".")) for i in df['q2759']]
df['q2762'] = [float(str(i).replace(",", ".")) for i in df['q2762']]
df['q2766'] = [float(str(i).replace(",", ".")) for i in df['q2766']]
df['q2769'] = [float(str(i).replace(",", ".")) for i in df['q2769']]
df['q2772'] = [float(str(i).replace(",", ".")) for i in df['q2772']]
df['q2776'] = [float(str(i).replace(",", ".")) for i in df['q2776']]
df['q2779'] = [float(str(i).replace(",", ".")) for i in df['q2779']]
df['q2783'] = [float(str(i).replace(",", ".")) for i in df['q2783']]
df['q2786'] = [float(str(i).replace(",", ".")) for i in df['q2786']]
df['q2789'] = [float(str(i).replace(",", ".")) for i in df['q2789']]
df['q2792'] = [float(str(i).replace(",", ".")) for i in df['q2792']]
df['q2795'] = [float(str(i).replace(",", ".")) for i in df['q2795']]


##Individuals to number_members

df['q78'] = pd.to_numeric(df['q78']) #inland_ind
df['q79'] = pd.to_numeric(df['q79']) #costal_ind
df['q81'] = pd.to_numeric(df['q81']) #urban_ind
df['q82'] = pd.to_numeric(df['q82']) #rural_ind


##Companies  to numbers
df['q501'] = pd.to_numeric(df['q501']) #num_resilience_provision_key_companies q501
df['q503'] = pd.to_numeric(df['q503']) #num_resilience_ocupation_companies q503
df['q506'] = pd.to_numeric(df['q506']) #indirect_num_resilience_provision_key q506
df['q508'] = pd.to_numeric(df['q508'])  #indirect_num_resilience_ocupation q508
df['q509'] = pd.to_numeric(df['q509']) #indirect_num_person q509
df['q533'] = pd.to_numeric(df['q533']) #inland_companies q533 %
df['q534'] = pd.to_numeric(df['q534']) #costal_companies q534 %
df['q536'] = pd.to_numeric(df['q536'])  #urban_companies q536 %
df['q537'] = pd.to_numeric(df['q537'])  #rural_companies q537 %


##Countries
df['q958'] = pd.to_numeric(df['q958'])#num_protection_harm_countries q958
df['q961'] = pd.to_numeric(df['q961'])#indirect_num_resilience_provision_key_countries q961
df['q963'] = pd.to_numeric(df['q963'])#indirect_num_protection_harm_countries q963
df['q964'] = pd.to_numeric(df['q964'])#indirect_num_person_countries q964
df['q966'] = pd.to_numeric(df['q966'])#inland_countries q966
df['q967'] = pd.to_numeric(df['q967'])#costal_countries q967
df['q969'] = pd.to_numeric(df['q969'])#urban_countries q969
df['q970'] = pd.to_numeric(df['q970'])#rural_countries q970

##REGIONS
df['q1399'] = pd.to_numeric(df['q1399'])
df['q1400'] = pd.to_numeric(df['q1400'])
df['q1402'] = pd.to_numeric(df['q1402'])
df['q1403'] = pd.to_numeric(df['q1403'])

##CITIES
df['q1832'] = pd.to_numeric(df['q1832'])
df['q1833'] = pd.to_numeric(df['q1833'])
df['q1835'] = pd.to_numeric(df['q1835'])
df['q1836'] = pd.to_numeric(df['q1836'])

##NATURAL SYSTEMS

df['q2268'] = pd.to_numeric(df['q2268'])
df['q2269'] = pd.to_numeric(df['q2269'])
df['q2271'] = pd.to_numeric(df['q2271'])
df['q2272'] = pd.to_numeric(df['q2272'])



#RA Subset
df_ra_general = df.dropna(subset=['q2712','q2741','q2748','q2755','q2765','q2775', 'q2782'])
df_ra_all = df.dropna(subset=['q2713','q2716','q2742',
'q2745','q2749','q2752',
'q2756','q2759','q2762',
'q2766','q2769','q2772',
'q2776','q2779','q2783',
'q2786','q2789','q2792','q2795'])



#Aggregate Information
#Key variables

total_individuals = df['q65'] #Individuals
number_p = df['q1']
number_members = df['q16'] #Members
num_countries = df['q959'] #Countries
num_companies = df['q504']  #COMPANIES
num_natural_system = df['q2296'] #Natural SYSTEM
num_cities = df['q1825']  #Cities
num_regions = df['q1392']  #REGIONS


#Título
image = Image.open('R2R_RGB_PINK.png')

col_x, col_y, col_z = st.columns(3)
col_x.image(image, width= 200)
#col_y.subheader("R2R DASHBOARD")
#col_y.caption("R2R REPORTING TOOL RESULTS - Beta version 2022/11/02")
#col_y.caption("Information from R2R Surveys reported until 2022/10/20")

#st.title("R2R METRICS DASHBOARD")
col_y.title("DATA EXPLORER")
col_y.write('Trial Phase')
#col_x.caption("Information from R2R Surveys until 2022/10/20")
st.markdown('<div style="text-align: justify;"></div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: justify;">R2R is a worldwide Campaign launched in 2021 by the High-Level Champions (HLT). It aims to increase resilience for four billion people living in vulnerable communities in collaboration with partner organizations from around the world while developing tools to support them in their work. The Campaign has developed a people-centred resilience Metrics Framework for non-state actors to report climate resilience actions and to quantify and validate their impact under a common framework. This framework will be opened to public consultation at COP27.</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: justify;"></div>', unsafe_allow_html=True) ##Espacio Texto
st.markdown('<div style="text-align: justify;">The Metrics Framework pursues several objectives at once, playing an essential function for both the Campaign and the global climate action and resilience communities. The Framework is the cornerstone of the R2R Campaign, serving as a guide for partners in taking action and for the HLC Team on how to manage and foster their work. It is composed of two complementary sets of metrics: Quantitative or Magnitude Metrics help estimate the effect size of the impact, fundamentally through the number of beneficiaries reached, and Qualitative or Depth Metrics, which help understand how the partners and their members are contributing to increasing resilience of people vulnerable to climate change, by observing on which key conditions (Resilience Attributes) are they making a change.</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: justify;"></div>', unsafe_allow_html=True) ##Espacio Texto

with st.expander("Summarised view of the Metrics Framework"):
    st.image("Summary_Metrics_R2R_Framework.png")

##st.markdown('Click *here* to download full metrics framework') ## Pendiente link

#st.markdown("""---""")

st.write(
    """
    <style>
    [data-testid="stMetricDelta"] svg {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,)

st.title("QUANTITATIVE / MAGNITUD METRICS")


st.markdown('<div style="text-align: justify;">‘Quantitative’ or ‘Magnitude’ metrics help estimate the size of the impact of R2R Actions in terms of the number of beneficiaries reached (linking up to the Campaign’s flagship ultimate goal of 4 billion people made more resilient)</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: justify;"></div>', unsafe_allow_html=True) ##Espacio Texto
with st.expander("Read more about R2R Target Beneficiaries"):
    st.write("""
        **R2R considers actions targeting six different beneficiaries**:
        - **Individuals**: actions that directly impact individuals, households or communities. This is, as explained above, the main target of the Campaign.
        - **Companies**: actions that make companies more resilient to climate change risks and, thus, indirectly protect livelihoods or the provision of key services.
        - **Cities, regions, countries**: actions that support local administrations in either protecting people from harm or ensuring they receive key services.
        - **Natural systems**: actions that protect, conserve or restore ecosystems to make them more resilient to climate change, thus protecting key ecosystem services.
        For all types of beneficiaries except individuals, households and communities, Partners are expected to provide an estimation of how many individuals would be indirectly benefited. Thus, metrics will be obtained for the direct impact of the R2R initiatives on each of these different kinds of beneficiaries, plus the indirect impact of all kinds of actions on individuals. Partners are asked to provide a robust methodology to justify this estimation.
    """)

st.subheader("R2R GLOBAL CAMPAIGN CURRENT STATUS")
st.caption("Information reported by R2R partners until 20/10/2022")

col1, col2, col3 = st.columns(3)

col1.metric("Total Nº of Partners",36)
col2.metric("Total Nº of Partners reporting Pledge",number_p.count())
col3.metric("Total Nº of Partners reporting Plan", 19)

col1.metric("Total Nº of Plans reported", 24,"(out of 19 partners)")
col2.metric("Total Nº of Members Pledged", int(number_members.sum()),"(out of 23 partners)")##
col3.metric("Countries in which they operate",175,"(out of 19 partners)")

col1.metric("Total Nº Invididuals Pledged",numerize(int(total_individuals.sum())*0.75),"(out of 17 partners)")
#col2.metric("Total Nº Companies Pledged",int(num_companies.sum()*0.75),"(out of 5 partners)")
col2.metric("Total Nº Cities Pledged",int(num_cities.sum()*0.75),"(out of 5 partners)")
col3.metric("Total Nº Region Pledged",int(num_regions.sum()*0.75),"(out of 5 partners)")
col1.metric("Total Nº Hectares Natural Systems Pledged",numerize(int(num_natural_system.sum())*0.75),"(out of 3 partners)")

#col3.metric("Total Nº Countries Pledged",int(num_countries.sum()*0.75),"(out of 5 partners)")

#Radialgraph
#RA General

eqt_inc = df_ra_general['q2712'].mean()
pp_pl = df_ra_general['q2741'].mean()
learn = df_ra_general['q2748'].mean()
agen = df_ra_general['q2755'].mean()
social_coll = df_ra_general['q2765'].mean()
div_redun = df_ra_general['q2775'].mean()
assets_ra = df_ra_general['q2782'].mean()

df2 = pd.DataFrame(dict(
    r=[eqt_inc, pp_pl, learn, agen, social_coll, div_redun, assets_ra],
    theta=['Equity & Inclusivity','Preparedness & planning','Learning',
        'Agency', 'Social Collaboration','Flexibility','Assets']))
fig_ra_general = px.line_polar(df2, r='r', theta='theta', line_close=True, title="RESILIENCE ATTRIBUTES - R2R GLOBAL CAMPAIGN*")
fig_ra_general.update_traces(fill='toself')

#RA All

Equity=df_ra_all['q2713'].sum()
Inclusivity=df_ra_all['q2716'].sum()
Preparedness=df_ra_all['q2742'].sum()
Planning=df_ra_all['q2745'].sum()
Experential_learning=df_ra_all['q2749'].sum()
Educational_learning=df_ra_all['q2752'].sum()
Autonomy=df_ra_all['q2756'].sum()
Leardership=df_ra_all['q2759'].sum()
Decision_making=df_ra_all['q2762'].sum()
Collective_participation=df_ra_all['q2766'].sum()
Connectivity=df_ra_all['q2769'].sum()
Networking=df_ra_all['q2772'].sum()
Diversity=df_ra_all['q2776'].sum()
Redundancy=df_ra_all['q2779'].sum()
Finance=df_ra_all['q2783'].sum()
Natural_resources=df_ra_all['q2786'].sum()
Technologies=df_ra_all['q2789'].sum()
Infrastructure=df_ra_all['q2792'].sum()
Services=df_ra_all['q2795'].sum()


df3 = pd.DataFrame(dict(
    r=[Equity, Inclusivity,Preparedness,Planning,Experential_learning, Educational_learning,
    Autonomy, Leardership,Decision_making,Collective_participation,Connectivity,Networking,Diversity,
     Redundancy,Finance,Natural_resources,Technologies,Infrastructure,Services],
    theta=['Equity','Inclusivity','Preparedness','Planning','Experential learning','Educational learning',
    'Autonomy','Leardership','Decision making','Collective participation','Connectivity','Networking','Diversity ',
    'Redundancy','Finance','Natural resources','Technologies','Infrastructure',
    'Services']))
fig_ra_all = px.line_polar(df3, r='r', theta='theta', line_close=True,title="RESILIENCE ATTRIBUTE'S SUBCATEGORIES - R2R GLOBAL CAMPAIGN*")
fig_ra_all.update_traces(fill='toself')

#Priority groups
#subset=['q19','q20','q21','q22','q23','q24','q25','q26','q27','q28'])
columns_pg = list(range(18,27)) #Other Excluded!
df_p_g  = df.iloc[:,columns_pg ]
df_p_g = df_p_g.stack().value_counts()
 #(normalize=True)
fig_p_g=px.bar(df_p_g,title="PRIORITY GROUPS (frequency)*",height=500, width=500, orientation='h')
fig_p_g.update_layout(showlegend=False)



#Macro REGIONS
#subset=['q30','q31','q32','q33','q34','q35','q36','q37','q38','q39','q40'])
#columns_m_r = list(range(29,39)) #Other Excluded!
#df_m_r  = df.iloc[:,columns_m_r ]
#df_m_r = df_m_r.stack().value_counts() #(normalize=True)
#fig_m_r=px.bar(df_m_r,title="Macro Region (frequency)",height=500, width=500, orientation='h')
#fig_m_r.update_layout(showlegend=False)
#st.write(fig_m_r)
#st.write(df_m_r.count())

#Key areas
#subset=['q42','q43','q44','q45','q46','q47','q48'])
columns_k_a = list(range(41,47)) #Other Excluded!
df_k_a  = df.iloc[:,columns_k_a ]
df_k_a = df_k_a.stack().value_counts(normalize=True) #(normalize=True)
df_k_a = df_k_a.reset_index() ## PENDIENTE: CAMBIAR NOMBRE A COLUMNAS PARA MEJORAR VISUALIZACIÓN
fig_k_a = px.pie(df_k_a, title="KEY AREA OF ACTION (%)*",height=500, width=500, values=0, names= 'index')
#fig_k_a.update_layout(legend=dict(
#    yanchor="top",
#    y=0.0,
#    xanchor="left",
#    x=0.0))


tab1, tab2, tab3 = st.tabs(["MAP OF COUNTRIES IN WHICH THEY OPERATE", "PRIORITY GROUPS", "KEY AREAS OF ACTION"])

with tab1:
   st.markdown("**MAP OF COUNTRIES IN WHICH THEY OPERATE***")
   st.image("heatmap_country_pledged.png", width=900)
   st.write("*Note: Frequency of countries regarding where partners operate, considering all possible target beneficiaries (darkness means that more partners operate in those countries)")

with tab2:
   st.write(fig_p_g)

with tab3:
   st.write(fig_k_a)


st.title("QUALITATIVE / DEPTH METRICS")

###DEFINITIONS DICTIONARY
df_ra_def = pd.read_csv('Definiciones_RA.csv',sep=';')
#df_ra_def = df.dropna(how = 'all')
#st.write(df_ra_def)

df_sub_ra_def = pd.read_csv('Definiciones_sub_RA_CSV.csv',sep=';')
#df_sub_ra_def = df.dropna(how = 'all')
#st.write(df_ra_def)

st.markdown('<div style="text-align: justify;"> ‘Qualitative’ or ‘Depth’ metrics help understand how the partners and their members contribute to increasing the resilience of people vulnerable to climate change by observing which key conditions (seven Resilience Attributes) are making a change. Resilience attributes act as an intermediary between the outcome of actions and increased resilience. The scientific literature acknowledges those Resilience Attributes to foster resilience or empower resilience-driving transformations.These seven Resilience Attributes cover most of the aspects of resilience building for the initiatives across the three constituting dimensions of resilience (plan, cope and learn). They are operationalized through 19 subcategories that address different aspects of the definitions of their correspondent Resilience Attribute.</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: justify;"></div>', unsafe_allow_html=True)

with st.expander("Summary of the seven Resilience Attributes and their nineteen subcategories"):
    st.image("Resilience_Atributes_Summary.png")


st.subheader("R2R GLOBAL CAMPAIGN RESILIENCE ATTRIBUTES AND SUB-CATEGORIES")
st.caption("Information from R2R Surveys until 2022/10/20")

col_a, col_b = st.columns(2)

col_a.write(fig_ra_general)
col_a.caption("*Out of 21 Partners")
ra_dictionary = col_a.selectbox(
    "SELECT DESCRIPTION OF A RESILIENCE ATTRIBUTE:",
    options=df_ra_def["RA"].unique())
df_ra_def_sel = df_ra_def.query('RA == @ra_dictionary')
col_a.markdown(str(df_ra_def_sel['RA'].values[0])+": "+str(df_ra_def_sel['DEF'].values[0]))
col_a.markdown("**SUBCATEGORIES**: "+str(df_ra_def_sel['SUBRA'].values[0])+".")

col_b.write(fig_ra_all)
col_b.caption("*Out of 21 Partners")
sub_ra_dictionary = col_b.selectbox(
        "SELECT DESCRIPTION OF RESILIENCE ATTRIBUTE'S SUB-CATEGORY:",
    options=df_sub_ra_def['SUB_RA'].unique())

df_sub_ra_def_sel = df_sub_ra_def.query('SUB_RA == @sub_ra_dictionary')
col_b.markdown(str(df_sub_ra_def_sel['SUB_RA'].values[0])+":  "+df_sub_ra_def_sel['DEF_SUB'].values[0])



### INFORMATION BY PARTNER
#FILTER SelectBox por partner
st.markdown("""---""")

st.title("INFORMATION BY PARTNER")

partner = st.selectbox(
    "SELECT R2R PARTNER: 🔎 ",
    options=df["q1"].unique())
df_select = df.query('q1 == @partner')


#Key variables SELECTION

s_total_individuals = df_select['q65'] #Individuals
s_number_members = df_select['q16'] #Members
s_num_countries = df_select['q959'] #Countries
s_num_companies = df_select['q504']  #COMPANIES
s_num_natural_system = df_select['q2296'] #Natural SYSTEM
s_num_cities = df_select['q1825']  #Cities
s_num_regions = df_select['q1392']  #REGIONS

s_inland_ind = df_select['q78']
s_costal_ind = df_select['q79']
s_urban_ind = df_select['q81']
s_rural_ind = df_select['q82']

s_num_resilience_provision_key_companies = df_select['q501']
s_num_resilience_ocupation_companies= df_select['q503']
s_indirect_num_resilience_provision_key= df_select['q506']
s_indirect_num_resilience_ocupation= df_select['q508']
s_indirect_num_person= df_select['q509']
s_inland_companies = df_select['q533']
s_costal_companies = df_select['q534']
s_urban_companies = df_select['q536']
s_rural_companies = df_select['q537']


s_num_resilience_provision_key_countries = df_select['q956']
s_num_protection_harm_countries = df_select['q958']
s_inland_countries = df_select['q966']
s_costal_countries = df_select['q967']
s_urban_countries = df_select['q969']
s_rural_countries = df_select['q970']

s_inland_region = df_select['q1399']
s_costal_region = df_select['q1400']
s_urban_region = df_select['q1402']
s_rural_region = df_select['q1403']

s_inland_cities = df_select['q1832']
s_costal_cities = df_select['q1833']
s_urban_cities = df_select['q1835']
s_rural_cities = df_select['q1836']

s_inland_natsys  = df_select['q2268']
s_costal_natsys = df_select['q2269']
s_urban_natsys = df_select['q2271']
s_rural_natsys = df_select['q2272']



p_priority_g_list = df_select['q29'].values[0]
p_key_area_action_list = df_select['q49'].values[0]
p_macroregion_g_list = df_select['q41'].values[0]
p_engagement_scope_list = df_select['q63'].values[0]

p_ind_hazard= df_select['q103'].values[0]
p_ind_continents= df_select['q110'].values[0]

p_type_resilience_companies = df_select['q499'].values[0]
p_sector_companies = df_select['q531'].values[0]
p_hazard_companies = df_select['q558'].values[0]
p_continents_companies = df_select['q565'].values[0]

p_hazard_countries= df_select['q991'].values[0]
p_type_resilience_countries= df_select['q954'].values[0]
p_continents_countries= df_select['q998'].values[0]

p_hazard_region = df_select['q1424'].values[0]
p_type_resilience_region = df_select['q1387'].values[0]
p_continents_region = df_select['q1431'].values[0]

p_hazard_cities = df_select['q1857'].values[0]
p_type_resilience_cities = df_select['q1820'].values[0]
p_continents_cities = df_select['q1864'].values[0]


p_type_nansyst = df_select['q2262'].values[0]
p_virg_antrho_nansyst = df_select['q2266'].values[0]
p_hazards_nansyst = df_select['q2293'].values[0]
p_continents_natsys = df_select['q2304'].values[0]



#Decription of partner by filter

p_short_name = df_select['q2'].values[0]
p_description = df_select['q4'].values[0]
st.markdown(p_short_name+" - "+p_description)
p_admition_date = df_select['q3'].values[0]

col1, col2 = st.columns(2)
col1.markdown("Admission Date: "+p_admition_date)
p_office_city = df_select['q9'].values[0]
p_office_country = df_select['q12'].values[0]
p_web = df_select['q15'].values[0]
col1.markdown("Location of the headquarters: "+p_office_city+", "+p_office_country)
col1.markdown(p_web)
col2.metric("Number of Members", int(s_number_members.sum()))



#st.markdown("""---""")

col1, col2, col3 = st.columns(3)
col1.markdown('**PRIORITY GROUPS**')
col1.markdown(p_priority_g_list)

col2.markdown('**AREA WHERE THEY OPERATE**')
col2.markdown(p_key_area_action_list)

col3.markdown('**WHERE THEY OPERATE**')
col3.markdown(p_macroregion_g_list)

st.markdown("""---""")

engagement_list = list(range(56,62)) #Other Excluded!
df_engagement = df_select.iloc[:,engagement_list]
df_engagement  = df_engagement.stack().value_counts() #(normalize=True)
df_engagement = df_engagement.reset_index()
#st.write(df_engagement.head())


col1, col2 = st.columns(2)
col1.title("PLEDGE")
col2.caption('')
col2.subheader(str(partner))

st.subheader('MAGNITUD METRICS')

engagement = st.selectbox(
    "SELECT -"+str.upper(p_short_name)+"-  TARGET BENEFICIARIES PLEDGED :",
        options= df_engagement['index'].unique())
df_select_engagement = df_select.query('index == @engagement')  ####REVISAR


col1, col2, col3, col4,col5 = st.columns(5)

if engagement == 'Individuals':
    col1.metric("Total Pledge Direct Beneficiaries",numerize(s_total_individuals.sum()))
    col2.metric("Inland (%)*",s_inland_ind) 
    col3.metric("Costal (%)*",s_costal_ind)           
    col4.metric("Urban (%)*",s_urban_ind) 
    col5.metric("Rural (%)*",s_rural_ind)
    st.markdown("**Hazards to provide Resilience**: "+p_ind_hazard)
    st.markdown("**Continents where they operate**: "+p_ind_continents)
    st.caption("*estimated information")


elif engagement == 'Companies':
    col1.metric("Total Pledge Nº Companies",numerize(s_num_companies.sum()))
    col2.metric("Inland (%)*",int(s_inland_companies))
    col3.metric("Costal (%)*",int(s_costal_companies))
    col4.metric("Urban (%)*",int(s_urban_companies))
    col5.metric("Rural (%)*",int(s_rural_companies))

    #st.subheader("Sector Companies")
    st.markdown("**Sector Companies**: "+p_sector_companies)
    st.markdown("**Hazard Companies**: "+p_hazard_companies)
    st.markdown(p_hazard_companies)
    #col1.metric("Number Resilience - Provision Key",s_num_resilience_provision_key_companies)
    #col1.metric("Number Resilience - Ocupation Companies",s_num_resilience_ocupation_companies)
    #st.subheader("**Type Resilience Companies**: "+p_type_resilience_companies)
    st.markdown("**Type Resilience Companies**: "+p_type_resilience_companies)
    #st.subheader("Continents Companies")
    st.markdown("**Continents**: "+p_continents_companies)
    #col1.metric("Indirect - Resilience provition key)",s_indirect_num_resilience_provision_key)
    #col1.metric("Indirect - Resilience ocupation)",s_indirect_num_resilience_ocupation)
    #col1.metric("Indirect - Number of person Beneficiaries",s_indirect_num_person)
    st.caption("*estimated information")

elif engagement == 'Countries':
    col1.metric("Number of Countries where they operate",int(s_num_countries.sum()))
    col2.metric("Inland (%)*",int(s_inland_countries))
    col3.metric("Costal (%)*",int(s_costal_countries))
    col4.metric("Urban (%)*",int(s_urban_countries))
    col5.metric("Rural (%)*",int(s_rural_countries))
    st.markdown("**Hazards countries**:"+p_hazard_countries)
    #st.markdown(p_hazard_countries)
    st.markdown("**Type of Resilience**: "+p_type_resilience_countries)
    #st.markdown(p_type_resilience_countries)
    st.markdown("**Continents**: "+p_continents_countries)
    #st.markdown(p_continents_countries)
    #s_num_resilience_provision_key_countries= df_plan_select['q956']
    #s_num_protection_harm_countries= df_plan_select['q958']
    st.caption("*estimated information")

elif engagement == 'Regions':
    col1.metric("Total Pledge Region",numerize(s_num_regions.sum()))
    col2.metric("Inland (%)*",s_inland_region)
    col3.metric("Costal (%)*",s_costal_region)
    col4.metric("Urban (%)*",s_urban_region)
    col5.metric("Rural (%)*",s_rural_region)

    #st.subheader("Hazards Region")
    st.markdown("**Hazards Region**: "+p_hazard_region)
    #st.subheader("Type of Resilience")
    st.markdown("**Type of Resilience**: "+p_type_resilience_region)
    #st.subheader("Continents of Region")
    st.markdown("**Continents**: "+p_continents_region)
    #s_num_resilience_provision_key_countries= df_plan_select['q956']
    #s_num_protection_harm_countries= df_plan_select['q958']
    st.caption("*estimated information")

elif engagement == 'Cities':
    col1.metric("Total Pledge Cities",numerize(s_num_cities.sum()))
    col2.metric("Inland (%)*",s_inland_cities)
    col3.metric("Costal (%)*",s_costal_cities)
    col4.metric("Urban (%)*",s_urban_cities)
    col5.metric("Rural (%)*",s_rural_cities)

    #st.subheader("Hazards Cities")
    st.markdown("**Hazards Cities**: "+p_hazard_cities)
    #st.subheader("Type of Resilience")
    st.markdown("**Type of Resilience**: "+p_type_resilience_cities)
    #st.subheader("Continents of countries")
    st.markdown("**Continents**: "+p_continents_cities)

    st.caption("*estimated information")

elif engagement == 'Natural Systems':
    col1.metric("Total Hectares Natural Systems",numerize(s_num_natural_system.sum()))
    col2.metric("Inland (%)*",s_inland_natsys)
    col3.metric("Costal (%)*",s_costal_natsys)
    col4.metric("Urban (%)*",s_urban_natsys)
    col5.metric("Rural (%)*",s_rural_natsys)
    st.markdown("**Type of Natural System**: "+p_type_nansyst)
    st.markdown("**Virgin or Anthropically**: "+p_virg_antrho_nansyst)
    st.markdown("**Hazards**: "+p_hazards_nansyst)
    st.markdown("**Continents**: "+str(p_continents_natsys))
    st.caption("*estimated information")

else:
    st.caption("")



#st.markdown("""---""")

#Radialgraph

#RA General

s_eqt_inc = df_select['q2712'].mean()
s_pp_pl = df_select['q2741'].mean()
s_learn = df_select['q2748'].mean()
s_agen = df_select['q2755'].mean()
s_social_coll = df_select['q2765'].mean()
s_div_redun = df_select['q2775'].mean()
s_assets_ra = df_select['q2782'].mean()

s_df2 = pd.DataFrame(dict(
    r=[s_eqt_inc, s_pp_pl, s_learn, s_agen, s_social_coll, s_div_redun, s_assets_ra],
    theta=['Equity & Inclusivity','Preparedness &  Planning','Learning',
        'Agency', 'Social Collaboration','Flexibility','Assets']))
s_fig_ra_general = px.line_polar(s_df2, r='r', theta='theta', line_close=True, title="RESILIENCE ATTRIBUTES - "+str(p_short_name))
s_fig_ra_general.update_traces(fill='toself')


#RA All

s_Equity=df_select['q2713'].sum()
s_Inclusivity=df_select['q2716'].sum()
s_Preparedness=df_select['q2742'].sum()
s_Planning=df_select['q2745'].sum()
s_Experential_learning=df_select['q2749'].sum()
s_Educational_learning=df_select['q2752'].sum()
s_Autonomy=df_select['q2756'].sum()
s_Leardership=df_select['q2759'].sum()
s_Decision_making=df_select['q2762'].sum()
s_Collective_participation=df_select['q2766'].sum()
s_Connectivity=df_select['q2769'].sum()
s_Networking=df_select['q2772'].sum()
s_Diversity=df_select['q2776'].sum()
s_Redundancy=df_select['q2779'].sum()
s_Finance=df_select['q2783'].sum()
s_Natural_resources=df_select['q2786'].sum()
s_Technologies=df_select['q2789'].sum()
s_Infrastructure=df_select['q2792'].sum()
s_Services=df_select['q2795'].sum()


s_df3 = pd.DataFrame(dict(
    r=[s_Equity, s_Inclusivity,s_Preparedness,s_Planning,s_Experential_learning, s_Educational_learning,
    s_Autonomy, s_Leardership,s_Decision_making,s_Collective_participation,s_Connectivity,s_Networking,s_Diversity,
     s_Redundancy,s_Finance,s_Natural_resources,s_Technologies,s_Infrastructure,s_Services],
    theta=['Equity','Inclusivity','Preparedness','Planning','Experential Learning','Educational Learning',
    'Autonomy','Leardership','Decision making','Collective participation','Connectivity','Networking','Diversity ',
    'Redundancy','Finance','Natural resources','Technologies','Infrastructure',
    'Services']))
s_fig_ra_all = px.line_polar(s_df3, r='r', theta='theta', line_close=True,title="RESILIENCE ATTRIBUTE'S SUB-CATEGORIES - "+str(p_short_name))
s_fig_ra_all.update_traces(fill='toself')


Equity_d = df_select['q2714'].values[0]
Inclusivity_d = df_select['q2739'].values[0]
Preparedness_d = df_select['q2743'].values[0]
Planning_d = df_select['q2746'].values[0]
Experential_learning_d = df_select['q2750'].values[0]
Educational_learning_d = df_select['q2753'].values[0]
Autonomy_d = df_select['q2757'].values[0]
Leardership_d = df_select['q2760'].values[0]
Decision_making_d = df_select['q2763'].values[0]
Collective_participation_d = df_select['q2767'].values[0]
Connectivity_d = df_select['q2770'].values[0]
Networking_d = df_select['q2773'].values[0]
Diversity_d = df_select['q2777'].values[0]
Redundancy_d = df_select['q2780'].values[0]
Finance_d = df_select['q2784'].values[0]
Natural_resources_d = df_select['q2787'].values[0]
Technologies_d = df_select['q2790'].values[0]
Infrastructure_d = df_select['q2793'].values[0]
Services_d = df_select['q2796'].values[0]


st.subheader('DEPTH METRICS')
col1, col2 = st.columns(2)
col1.write(s_fig_ra_general)
col2.write(s_fig_ra_all)

sub_ra_partner = st.selectbox(
        "SELECT A RESILIENCE ATTRIBUTE'S SUB-CATEGORY DESCRIPTION BY "+str(p_short_name),
    options=df_sub_ra_def['SUB_RA'].unique())

if sub_ra_partner == 'Equity':
    st.markdown("Equity: "+Equity_d)
elif sub_ra_partner == 'Inclusivity':
    st.markdown("Inclusivity: "+Inclusivity_d)
elif sub_ra_partner == 'Preparedness':
    st.markdown("Preparedness: "+Preparedness_d)
elif sub_ra_partner == 'Planning':
    st.markdown("Planning: "+Planning_d)
elif sub_ra_partner == 'Experential learning':
    st.markdown("Experential learning: "+Experential_learning_d)
elif sub_ra_partner == 'Educational learning':
    st.markdown("Educational learning: "+Educational_learning_d)
elif sub_ra_partner == 'Autonomy':
    st.markdown("Autonomy: "+Autonomy_d)
elif sub_ra_partner == 'Leardership':
    st.markdown("Leardership: "+Leardership_d)
elif sub_ra_partner == 'Decision making':
    st.markdown("Decision making: "+Decision_making_d)
elif sub_ra_partner == 'Collective participation':
    st.markdown("Collective participation: "+Collective_participation_d)
elif sub_ra_partner == 'Connectivity':
    st.markdown("Connectivity: "+Connectivity_d)
elif sub_ra_partner == 'Networking':
    st.markdown("Networking: "+Networking_d)
elif sub_ra_partner == 'Diversity ':
    st.markdown("Diversity: "+Diversity_d)
elif sub_ra_partner == 'Redundancy':
    st.markdown("Redundancy: "+Redundancy_d)
elif sub_ra_partner == 'Finance ':
    st.markdown("Finance: "+Finance_d)
elif sub_ra_partner == 'Natural resources':
    st.markdown("Natural resources: "+Natural_resources_d)
elif sub_ra_partner == 'Technologies':
    st.markdown("Technologies: "+Technologies_d)
elif sub_ra_partner == 'Infrastructure':
    st.markdown("Infrastructure: "+Infrastructure_d)
elif sub_ra_partner == 'Services':
    st.markdown("Services: "+Services_d)


#st.markdown("[Select another partner](#INFORMATION-BY-PARTNER)")
#st.markdown("[Section 1](#section-1)")

st.markdown("""---""")


col1, col2 = st.columns(2)
#st.title("R2R METRICS DASHBOARD")
col1.title("PLAN")
#st.markdown('Short Description of the Survey "Plan"')

#data
df_plan = pd.read_csv('BD_PLAN_CSV_08_nov.csv',sep=';', header=None, prefix="p").iloc[2:]
df_plan.set_index("p0", inplace = True)
df_plan = df_plan.dropna(how = 'all')

##Crear un filtro segun partner

#partner = st.selectbox(
#    "Select a partner",
#    options=df_plan["p2"].unique())
df_plan_partner = df_plan.query('p2 == @partner')


#number_of_plan = str((df_plan_partner['p2788'].value_counts()).sum())
number_of_plan = sum(pd.to_numeric(df_plan_partner['p2788']))

partner_name = str(partner)

col2.caption('')
col2.subheader(partner+" has "+str(number_of_plan)+" plan(s) reported")

plan = st.selectbox(
    "SELECT A PLAN:",
    options=df_plan_partner["p4"].unique())
df_plan_select = df_plan.query('p4 == @plan') ##PENDIENTE. QUÉ HACER CUANDO PARTNERS NO TIENEN PLAN!!

#Short description of plan
Description_Plan = df_plan_select['p2774'].values[0]  #Description_Plan
st.markdown("**Description of plan**: "+Description_Plan)

star_date = df_plan_partner['p2776'].values[0]
if len(str(star_date)) > 3:
    st.markdown("**Start Date**: "+str(star_date))

Timeline_of_impact = df_plan_select['p174'].values[0]  #Timeline_of_impact
enduring_not_enduring = df_plan_select['p175'].values[0]  #enduring_not_enduring
who_long_action = df_plan_select['p176'].values[0]  #who_long_action

if len(str(Timeline_of_impact)) > 3:
    st.markdown("**Finish Date**: "+Timeline_of_impact+" ("+str(enduring_not_enduring)+" Action)")
if who_long_action == "Fading":
    st.caption("*Date in case Fading action :"+str(who_long_action))


t_a_domain = df_plan_select['p16'].values[0]  #t_a_domain
t_a_assessments_monitoring = df_plan_select['p25'].values[0]  #t_a_assessments_monitoring
t_a_early_action = df_plan_select['p30'].values[0]  #t_a_early_action
t_a_preparedness = df_plan_select['p37'].values[0]  #t_a_preparedness
t_a_governance_capacity_building = df_plan_select['p67'].values[0]  #t_a_governance_capacity_building
t_a_nature_based_solutions = df_plan_select['p82'].values[0]  #t_a_nature_based_solutions
t_a_infrastructure_services = df_plan_select['p107'].values[0]  #t_a_infrastructure_services
t_a_risk_transfer = df_plan_select['p117'].values[0]  #t_a_risk_transfer
t_a_sharing_best_practice = df_plan_select['p142'].values[0]  #t_a_sharing_best_practice
t_a_increasing_finance = df_plan_select['p152'].values[0]  #t_a_increasing_finance
t_a_other = df_plan_select['p15'].values[0]  #t_a_other


#st.subheader("ACTION AREA")
#st.markdown(str(t_a_domain))

st.subheader("TYPE OF ACTION")

if len(str(t_a_assessments_monitoring)) > 3:
    st.markdown("**CLIMATE RISK VULNERABILITY ASSESSMENTS; DISCLOSURE & MONITORING**: "+str(t_a_assessments_monitoring)+".")

if len(str(t_a_early_action)) > 3:
    st.markdown("**EARLY WARNING SYSTEMS & EARLY ACTION**:"+str(t_a_early_action)+".")

if len(str(t_a_preparedness)) > 3:
    st.markdown("**PREPAREDNESS, CONTINGENCY PLANS/ EMERGENCY RESPONSE**:"+str(t_a_preparedness)+".")

if len(str(t_a_governance_capacity_building)) > 3:
    st.markdown("**CLIMATE RISK GOVERNANCE & CAPACITY-BUILDING**: "+str(t_a_governance_capacity_building)+".")

if len(str(t_a_nature_based_solutions)) > 3:
    st.markdown("**NATURE-BASED SOLUTIONS TO REDUCE RISKS**: "+str(t_a_nature_based_solutions)+".")

if len(str(t_a_infrastructure_services)) > 3:
    st.markdown("**CLIMATEPROOFING INFRASTRUCTURE & SERVICES**: "+str(t_a_infrastructure_services)+".")

if len(str(t_a_risk_transfer)) > 3:
    st.markdown("**RISK TRANSFER: INSURANCE & SOCIAL PROTECTION**: "+str(t_a_risk_transfer)+".")

if len(str(t_a_sharing_best_practice)) > 3:
    st.markdown("**SHARING BEST PRACTICE ON CLIMATE RISK ACTIONS MANAGEMENT**: "+str(t_a_sharing_best_practice)+".")

if len(str(t_a_increasing_finance)) > 3:
    st.markdown("**INCREASING THE VOLUME, QUALITY OF PUBLIC AND PRIVATE FINANCE**: "+str(t_a_increasing_finance)+".")

if len(str(t_a_other)) > 3:
    st.markdown("**OTHER**: "+str(t_a_other)+".")



st.subheader("CURRENT STATUS OF PROYECTS")
#df_plan_select['p2780'] = pd.to_numeric(df['p2780']) #Estimated_numer_of_proyects

col1, col2, col3, col4 = st.columns(4)


#Number of your members that participate in implementing these projects.
#df_plan_select['p2781'] = pd.to_numeric(df['p2781']) #Number_of_members_
Number_of_members_ = df_plan_select['p2787'].values[0]  #Number_of_members_
col1.markdown("NUMBER OF MEMBERS PARTICIPATING: ")
col1.subheader(str(Number_of_members_))
#st.metric("Number of members that participate in implementing these projects", int(Number_of_members_))


Estimated_numer_of_proyects = df_plan_select['p2786'].values[0]  #Estimated_numer_of_proyects
col2.markdown("NUMBER OF PROJECTS PLANNED*:")
col2.subheader(str(Estimated_numer_of_proyects))
#col1.caption("*e.g. a project can be a specific instance of the action in a given territory/with a given beneficiary")
#st.metric("Estimated number of proyects", int(Estimated_numer_of_proyects))

#Please indicate how many of the planned projects have started. *in %
#df_plan_select['p2777'] = pd.to_numeric(df['p2777']) #Projects_have_started
Projects_have_started = df_plan_select['p2777'].values[0]  #Projects_have_started
col3.markdown("PROYECTS ALREADY STARTED*:")
col3.subheader(str(Projects_have_started)+" %")
#st.metric("Number of proyects already started", int(Projects_have_started))

#Please indicate how many of the planned projects have been completed.* in %
#df_plan_select['p2778'] = pd.to_numeric(df['p2778']) #Projects_have_been_comleted
Projects_have_been_comleted = df_plan_select['p2778'].values[0]  #Projects_have_been_comleted
col4.markdown("PROYECTS COMPLETED*:")
col4.subheader(str(Projects_have_been_comleted)+" %")
#st.metric("Number of proyects completed", int(Projects_have_been_comleted))
st.caption("*estimated information")

# Type of Hazards
st.subheader("HAZARDS")
Hazard = df_plan_select['p173'].values[0]  #Hazard
st.markdown(str(Hazard))


st.subheader("TARGET BENEFICIARIES")

#Key variables SELECTION PLAN


engagement_plan_list = list(range(176,182)) #Other Excluded!
df_engagement_plan = df_plan_select.iloc[:,engagement_plan_list]
df_engagement_plan  = df_engagement_plan.stack().value_counts() #(normalize=True)
df_engagement_plan = df_engagement_plan.reset_index()

engagement_plan = st.selectbox(
    "SELECT -"+str.upper(p_short_name)+" -  TARGET BENEFICIARIES  PLAN:",
        options= df_engagement_plan['index'].unique())
df_plan_engamente_select = df_engagement_plan.query('index == @engagement_plan')


df_plan_select['p186'] = pd.to_numeric(df_plan_select['p186'])
s_total_individuals_plan = df_plan_select['p186'] #Individuals


s_num_countries_plan = pd.to_numeric(df_plan_select['p1087']) #Countries
s_num_companies_plan = pd.to_numeric(df_plan_select['p605'])  #COMPANIES
s_num_natural_system_plan = pd.to_numeric(df_plan_select['p2360']) #Natural SYSTEM
s_num_cities_plan = pd.to_numeric(df_plan_select['p1952'])  #Cities
s_num_regions_plan = pd.to_numeric(df_plan_select['p1499'])  #REGIONS

s_inland_ind_plan = pd.to_numeric(df_plan_select['p200'])
s_costal_ind_plan = pd.to_numeric(df_plan_select['p201'])
s_urban_ind_plan = pd.to_numeric(df_plan_select['p203'])
s_rural_ind_plan = pd.to_numeric(df_plan_select['p204'])


s_inland_companies_plan = pd.to_numeric(df_plan_select['p637'])
s_costal_companies_plan = pd.to_numeric(df_plan_select['p638'])
s_urban_companies_plan = pd.to_numeric(df_plan_select['p634'])
s_rural_companies_plan = pd.to_numeric(df_plan_select['p635'])



s_inland_countries_plan = pd.to_numeric(df_plan_select['p1094'])
s_costal_countries_plan = pd.to_numeric(df_plan_select['p1095'])
s_urban_countries_plan = pd.to_numeric(df_plan_select['p1097'])
s_rural_countries_plan = pd.to_numeric(df_plan_select['p1098'])

s_inland_region_plan = pd.to_numeric(df_plan_select['p1506'])
s_costal_region_plan = pd.to_numeric(df_plan_select['p1507'])
s_urban_region_plan = pd.to_numeric(df_plan_select['p1509'])
s_rural_region_plan = pd.to_numeric(df_plan_select['p1510'])

s_inland_cities_plan = pd.to_numeric(df_plan_select['p1959'])
s_costal_cities_plan = pd.to_numeric(df_plan_select['p1960'])
s_urban_cities_plan = pd.to_numeric(df_plan_select['p1962'])
s_rural_cities_plan = pd.to_numeric(df_plan_select['p1963'])

s_inland_natsys_plan  = pd.to_numeric(df_plan_select['p2378'])
s_costal_natsys_plan = pd.to_numeric(df_plan_select['p2379'])
s_urban_natsys_plan = pd.to_numeric(df_plan_select['p2381'])
s_rural_natsys_plan = pd.to_numeric(df_plan_select['p2382'])


p_ind_continents_plan= df_plan_select['p211'].values[0]

p_type_resilience_companies_plan = df_plan_select['p600'].values[0]
p_sector_companies_plan = df_plan_select['p627'].values[0]
p_continents_companies_plan = df_plan_select['p645'].values[0]

p_type_resilience_countries_plan= df_plan_select['p1082'].values[0] #1082
p_continents_countries_plan= df_plan_select['p1105'].values[0]

p_type_resilience_region_plan = df_plan_select['p1494'].values[0]
p_continents_region_plan = df_plan_select['p1517'].values[0]

p_type_resilience_cities_plan = df_plan_select['p1947'].values[0] #1947
p_continents_cities_plan = df_plan_select['p1970'].values[0]

p_type_nansyst_plan = df_plan_select['p2372'].values[0]
p_virg_antrho_nansyst_plan = df_plan_select['p2376'].values[0]

p_continents_natsys_plan = df_plan_select['p2389'].values[0]

col1, col2, col3,col4,col5 = st.columns(5)


if engagement_plan == 'Individuals':
    col1.metric("Total Plan Direct Beneficiaries",numerize(int(s_total_individuals_plan.sum())))
    col2.metric("Inland %*",s_inland_ind_plan)
    col3.metric("Costal %*",s_costal_ind_plan)
    col4.metric("Urban %*",s_urban_ind_plan)
    col5.metric("Rural %*",s_rural_ind_plan)
    st.markdown("**Continents where they operate**: "+p_ind_continents_plan)
    #st.markdown(p_ind_continents_plan)
    st.caption("*estimated information")


elif engagement_plan == 'Companies':
    col1.metric("Total Pledge Nº Companies",s_num_companies_plan.sum())
    col2.metric("Inland %*",s_inland_companies_plan)
    col3.metric("Costal %*",s_costal_companies_plan)
    col4.metric("Urban %*",s_urban_companies_plan)
    col5.metric("Rural %*",s_rural_companies_plan)
    st.markdown("**Sector Companies**: "+p_sector_companies_plan)
    #st.subheader("**Type Resilience Companies**: ")
    st.markdown("**Type Resilience Companies**: "+p_type_resilience_companies_plan)
    #st.subheader("**Continents Companies**: ")
    st.markdown("**Continents where they operate**: "+p_continents_companies_plan)
    st.caption("*estimated information")

elif engagement_plan == 'Countries':
    col1.metric("Number of Countries where they operate",s_num_countries_plan.sum())
    col2.metric("Inland %*",s_inland_countries_plan)
    col3.metric("Costal %*",s_costal_countries_plan)
    col4.metric("Urban %*",s_urban_countries_plan)
    col5.metric("Rural %*",s_rural_countries_plan)
    #st.subheader("**Type of Resilience**: ")
    st.markdown("**Type of Resilience**: "+p_type_resilience_countries_plan)
    #st.subheader("**Continents of countries**: ")
    st.markdown("**Continents where they operate**: "+p_continents_countries_plan)
    st.caption("*estimated information")


elif engagement_plan == 'Regions':
    col1.metric("Total Plan Region",s_num_regions.sum())
    col2.metric("Inland %*",s_inland_region_plan)
    col3.metric("Costal %*",s_costal_region_plan)
    col4.metric("Urban %*",s_urban_region_plan)
    col5.metric("Rural %*",s_rural_region_plan)
    #st.subheader("**Type of Resilience**:")
    st.markdown("**Type of Resilience**: "+p_type_resilience_region_plan)
    #st.subheader("**Continents of Region**: ")
    st.markdown("**Continents where they operate**: "+p_continents_region_plan)
    st.caption("*estimated information")

elif engagement_plan == 'Cities':
    col1.metric("Inland %*",s_inland_cities_plan)
    col2.metric("Costal %*",s_costal_cities_plan)
    col3.metric("Urban %*",s_urban_cities_plan)
    col4.metric("Rural %*",s_rural_cities_plan)
    #st.subheader("**Type of Resilience**: ")
    st.markdown("**Type of Resilience**: "+p_type_resilience_cities_plan)
    #st.subheader("**Continents of countries**: ")
    st.markdown("**Continents where they operates**: "+p_continents_cities_plan)
    st.caption("*estimated information")

elif engagement_plan == 'Natural System':
    col1.metric("Total Hectares Natural Systems",numerize(int(s_num_natural_system.sum())))
    col2.metric("Inland %*",s_inland_natsys_plan)
    col3.metric("Costal %*",s_costal_natsys_plan)
    col4.metric("Urban %*",s_urban_natsys_plan)
    col5.metric("Rural %*",s_rural_natsys_plan)
    st.markdown("**Type of Natural System**: "+str(p_type_nansyst_plan))
    #st.subheader("Virgin or Anthropically")
    st.markdown("**Virgin or Anthropically**: "+str(p_virg_antrho_nansyst_plan)) #q2266
    #st.subheader("Continents Natural Systems")
    st.markdown("**Continents where they operate**: "+str(p_continents_natsys_plan)) #q2304
    st.caption("*estimated information")

else:
    st.write("")

#st.markdown("""---""")
