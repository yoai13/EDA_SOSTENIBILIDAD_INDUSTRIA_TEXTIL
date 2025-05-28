from sklearn.datasets import load_iris
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

print("hola")
df = pd.read_csv("../data/Sustainable_Fashion_Trends.csv")

#Elimino la columna Brand_ID
df1 = df.drop('Brand_ID', axis=1)

print(df1['Sustainability_Rating'])

print("Valores únicos en 'Sustainability_Rating' antes del reemplazo:\n", df1['Sustainability_Rating'].unique())

#Según la clasificación medioambiental A es Excelente, B es Muy Bueno, C es Bueno y D es Correcto, y pasan a ser 1, 2, 3 y 4.

#Defino un diccionario de reemplazo
replacement_dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4}

#Realizo todos los reemplazos a la vez usando el diccionario
df1['Sustainability_Rating'] = df1['Sustainability_Rating'].replace(replacement_dict);

#HIPÓTESIS 1:Las marcas con una calificación de sostenibilidad más alta tienden a tener una menor huella de carbono, menor uso de agua y menor producción de residuos.

#Análisis por grupos de calificación de sostenibilidad
grouped_data = df1.groupby('Sustainability_Rating')[['Carbon_Footprint_MT', 'Water_Usage_Liters', 'Waste_Production_KG']].mean()

#Gráfica 1: Promedio de Huella de Carbono por Calificación de Sostenibilidad
plt.figure(figsize=(6, 3))
plt.bar(grouped_data.index, grouped_data['Carbon_Footprint_MT'], color="#0f110f")
plt.xlabel('Calificación de Sostenibilidad')
plt.ylabel('Promedio de Huella de Carbono (MT)' , fontsize=8)
plt.title('Promedio de Huella de Carbono por Calificación de Sostenibilidad')
plt.xticks(grouped_data.index)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig("../images/promedio_carbon_sostenibilidad.jpg")

#Gráfica 2: Promedio de Uso de Agua por Calificación de Sostenibilidad
plt.figure(figsize=(6, 3))
plt.bar(grouped_data.index, grouped_data['Water_Usage_Liters'], color="#1db6d1")
plt.xlabel('Calificación de Sostenibilidad')
plt.ylabel('Promedio de Uso de Agua (Liters)' , fontsize=8)
plt.title('Promedio de Uso de Agua por Calificación de Sostenibilidad')
plt.xticks(grouped_data.index)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig("../images/promedio_agua_sostenibilidad.jpg")

#Gráfica 3: Promedio de Producción de Desperdicios por Calificación de Sostenibilidad
plt.figure(figsize=(6, 3))
plt.bar(grouped_data.index, grouped_data['Waste_Production_KG'], color="#925C0B")
plt.xlabel('Calificación de Sostenibilidad')
plt.ylabel('Promedio de Producción de Desperdicios (KG)', fontsize=8)
plt.title('Promedio de Producción de Desperdicios por Calificación de Sostenibilidad')
plt.xticks(grouped_data.index)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig("../images/promedio_desperdicios_sostenibilidad.jpg")


#Cálculo de correlaciones: Calcula la correlación entre la calificación de sostenibilidad y las métricas de impacto ambiental. Una correlación negativa indicaría que a mayor calificación de sostenibilidad, menor es el impacto ambiental.
correlation_carbon = df1['Sustainability_Rating'].corr(df1['Carbon_Footprint_MT'])
print("Correlación entre la Calificación de Sostenibilidad y la Huella de Carbono", correlation_carbon)

correlation_water = df1['Sustainability_Rating'].corr(df1['Water_Usage_Liters'])
print("Correlación entre la Calificación de Sostenibilidad y el Uso de Agua", correlation_water)

correlation_waste = df1['Sustainability_Rating'].corr(df1['Waste_Production_KG'])
print("Correlación entre la Calificación de Sostenibilidad y la Producción de Desperdicios", correlation_waste)

#Visualizo la correlación entre la calificación de sostenibilidad y las métricas de impacto ambiental
#Creo un DataFrame para facilitar la graficación
correlations_data = pd.DataFrame({
    'Métrica': ['Carbono', 'Agua', 'Desperdicios'],
    'Correlación': [correlation_carbon, correlation_water, correlation_waste]
})

#Gráfico de barras horizontales
plt.figure(figsize=(8, 4))
plt.barh(correlations_data['Métrica'], correlations_data['Correlación'], color=["#0f110f", "#1db6d1", "#925C0B"])
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')  # Línea de referencia en cero
plt.ylabel('Métrica de Impacto Ambiental')
plt.xlabel('Correlación con Calificación de Sostenibilidad')
plt.title('Correlación entre la Calificación de Sostenibilidad y Métricas de Impacto Ambiental')

# Añadir los valores de correlación al lado de las barras
for index, value in enumerate(correlations_data['Correlación']):
    plt.text(value, index, f'{value:.4f}', va='center', ha='left', color= "red")

#Salvo la imagen
plt.savefig("../images/correlacion_sostenibilidad_metricas")


#Análisis de la Hipótesis 1

#HIPÓTESIS 2:** *La calificación de sostenibilidad promedio de las marcas ha mejorado con el tiempo.

#Agrupo por 'Year' y calcular la calificación de sostenibilidad promedio
average_sustainability_by_year = df1.groupby('Year')['Sustainability_Rating'].mean()
#Muestro la calificación de sostenibilidad promedio por año
print("Calificación de sostenibilidad promedio por año:\n", average_sustainability_by_year)

#Visualización
plt.figure(figsize=(5, 3))
sns.lineplot(x=average_sustainability_by_year.index, y=average_sustainability_by_year.values, marker='o', color= "#33ff4f")
plt.xlabel('Año')
plt.ylabel('Calificación de Sostenibilidad Promedio', fontsize=8)
plt.title('Tendencia de la Calificación de Sostenibilidad Promedio a lo Largo del Tiempo')
plt.grid(True)
plt.tight_layout()
plt.savefig("../images/sostenibilidad_promedio_years")

#Análisis de la Hipótesis 2


#HIPÓTESIS 3: La huella de carbono promedio, el uso de agua y la producción de residuos por marca han disminuido con el tiempo debido a avances tecnológicos y una mayor conciencia ambiental.

#Agrupo por año y calculo el Promedio de cada métrica:
average_impact_by_year = df1.groupby('Year')[['Carbon_Footprint_MT', 'Water_Usage_Liters', 'Waste_Production_KG']].mean().reset_index()
print("Promedio de impacto por año:\n", average_impact_by_year)

#Visualizo la tendencia para cada métrica
#Gráfica 1: Tendencia del Promedio de Huella de Carbono a lo Largo del Tiempo
plt.figure(figsize=(5, 3))
plt.plot(average_impact_by_year['Year'], average_impact_by_year['Carbon_Footprint_MT'], marker='o', linestyle='-', color="#33ff4f")
plt.xlabel('Año')
plt.ylabel('Promedio de Huella de Carbono (MT)', fontsize=8)
plt.title('Tendencia del Promedio de Huella de Carbono a lo Largo del Tiempo')
plt.grid(True)
plt.tight_layout()
plt.savefig("../images/tendencia_carbon_por_año.jpg")

#Gráfica 2: Tendencia del Promedio de Uso de Agua a lo Largo del Tiempo
plt.figure(figsize=(5, 3))
plt.plot(average_impact_by_year['Year'], average_impact_by_year['Water_Usage_Liters'], marker='o', linestyle='-', color="#33ff4f")
plt.xlabel('Año')
plt.ylabel('Promedio de Uso de Agua (Liters)', fontsize=8)
plt.title('Tendencia del Promedio de Uso de Agua a lo Largo del Tiempo')
plt.grid(True)
plt.tight_layout()
plt.savefig("../images/tendencia_agua_por_año.jpg")

#Gráfica 3: Tendencia del Promedio de Producción de Desperdicios a lo Largo del Tiempo
plt.figure(figsize=(5, 3))
plt.plot(average_impact_by_year['Year'], average_impact_by_year['Waste_Production_KG'], marker='o', linestyle='-', color="#33ff4f")
plt.xlabel('Año')
plt.ylabel('Promedio de Producción de Desperdicios (KG)', fontsize=6)
plt.title('Tendencia del Promedio de Producción de Desperdicios a lo Largo del Tiempo')
plt.grid(True)
plt.tight_layout()
plt.savefig("../images/tendencia_desperdicios_por_año.jpg")

#Análisis de la Hipótesis 3


#HIPÓTESIS 4: La proporción de marcas que adoptan prácticas de fabricación eco-friendly y ofrecen programas de reciclaje ha aumentado con el tiempo.

#Calculo la Proporción Anual de Marcas con Prácticas Sostenibles:

#Calculo el número total de marcas por año
total_brands_by_year = df1.groupby('Year')['Brand_Name'].nunique()

#Calculo el número de marcas con fabricación eco-friendly por año
eco_friendly_counts = df1[df1['Eco_Friendly_Manufacturing'] == 'Yes'].groupby('Year')['Brand_Name'].nunique()

#Calculo el número de marcas con programas de reciclaje por año
recycling_counts = df1[df1['Recycling_Programs'] == 'Yes'].groupby('Year')['Brand_Name'].nunique()

#Calculo la proporción de marcas con fabricación eco-friendly por año
proportion_eco_friendly = (eco_friendly_counts / total_brands_by_year).fillna(0)

#Calculo la proporción de marcas con programas de reciclaje por año
proportion_recycling = (recycling_counts / total_brands_by_year).fillna(0)

print("Proporción de marcas con fabricación eco-friendly por año:\n", proportion_eco_friendly)
print("\nProporción de marcas con programas de reciclaje por año:\n", proportion_recycling)

#Visualizo las Tendencias:
plt.figure(figsize=(12, 6))
plt.plot(proportion_eco_friendly.index, proportion_eco_friendly.values, marker='o', label='Fabricación Eco-Friendly', color="#33ff4f" )
plt.plot(proportion_recycling.index, proportion_recycling.values, marker='o', label='Programas de Reciclaje', color="orange")
plt.xlabel('Año')
plt.ylabel('Proporción de Marcas')
plt.title('Proporción de Marcas con Prácticas Sostenibles a lo Largo del Tiempo')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../images/practicas_sostenibles_años.jpg")

#Análisis de la Hipótesis 4

#HIPÓTESIS 5:"La huella de carbono promedio, el uso de agua promedio y de la producción de residuos promedio varían significativamente entre países.

#Promedio de Huella de Carbono por países
promedio_carbono_por_pais = df1.groupby('Country')['Carbon_Footprint_MT'].mean()

print("Promedio de huella de carbono por país:")
print(promedio_carbono_por_pais)

#Creo el gráfico de tarta
plt.figure(figsize=(5, 5))
plt.pie(promedio_carbono_por_pais,
        labels=promedio_carbono_por_pais.index,
        autopct='%1.1f%%', #formateo el porcentaje que se mostrará en cada porción (un decimal seguido del símbolo de porcentaje)
        startangle=140)

plt.title('Promedio de Huella de Carbono por País')
plt.axis('equal')  #Aseguro que la tarta sea un círculo.
plt.savefig("../images/practicas_sostenibles_años.jpg")

#Promedio de agua
promedio_agua_por_pais = df1.groupby('Country')['Water_Usage_Liters'].mean()

print("Promedio de gasto de agua por país:")
print(promedio_agua_por_pais)

#Creo el gráfico de tarta
plt.figure(figsize=(5, 5))
plt.pie(promedio_agua_por_pais,
        labels=promedio_agua_por_pais.index,
        autopct='%1.1f%%', #formateo el porcentaje que se mostrará en cada porción (un decimal seguido del símbolo de porcentaje)
        startangle=140)

plt.title('Promedio de gasto de agua por País')
plt.axis('equal')  #Aseguro que la tarta sea un círculo.
plt.savefig("../images/practicas_sostenibles_agua_años.jpg")

#Promedio de Desperdicios
promedio_desperdicios_por_pais = df1.groupby('Country')["Waste_Production_KG"].mean()

print("Promedio de desperdicios por país:")
print(promedio_desperdicios_por_pais)

#Creo el gráfico de tarta
plt.figure(figsize=(5, 5))
plt.pie(promedio_desperdicios_por_pais,
        labels=promedio_desperdicios_por_pais.index,
        autopct='%1.1f%%', #formateo el porcentaje que se mostrará en cada porción (un decimal seguido del símbolo de porcentaje)
        startangle=140)

plt.title('Promedio de desperdicios por País')
plt.axis('equal')  #Aseguro que la tarta sea un círculo.
plt.savefig("../images/practicas_sostenibles_waste_años.jpg")

#Análisis de la Hipótesis 5

#HIPÓTESIS 6: Hay correlación significativamente entre la huella de carbono, el uso de agua y de la producción de residuos entre países

#Agrupando las métricas por País
#Calculo el promedio de las métricas de impacto ambiental por país
promedio_impacto_por_pais = df1.groupby('Country')[['Carbon_Footprint_MT', 'Water_Usage_Liters', 'Waste_Production_KG']].mean()

print("Promedio de Métricas de Impacto Ambiental por País:\n", promedio_impacto_por_pais)

#Calculo la matriz de correlación entre las métricas a nivel de país
matriz_correlacion_pais = promedio_impacto_por_pais.corr(method='pearson')

print("\nMatriz de Correlación entre Métricas de Impacto Ambiental a Nivel de País:\n", matriz_correlacion_pais)

#Visualización de la matriz de correlación con un mapa de calor
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_correlacion_pais, annot=True, cmap="Greens", fmt=".2f")
plt.title('Correlación entre Métricas de Impacto Ambiental Promedio por País')
plt.tight_layout()
plt.savefig("../images/practicas_sostenibles_waste_años.jpg")

#Analizo de correlación específica entre dos métricas
correlacion_carbono_agua = promedio_impacto_por_pais['Carbon_Footprint_MT'].corr(promedio_impacto_por_pais['Water_Usage_Liters'], method='pearson')
print(f"\nCorrelación entre Promedio de Huella de Carbono y Promedio de Uso de Agua por País: {correlacion_carbono_agua:.2f}")

correlacion_carbono_residuos = promedio_impacto_por_pais['Carbon_Footprint_MT'].corr(promedio_impacto_por_pais['Waste_Production_KG'], method='pearson')
print(f"Correlación entre Promedio de Huella de Carbono y Promedio de Producción de Residuos por País: {correlacion_carbono_residuos:.2f}")

correlacion_agua_residuos = promedio_impacto_por_pais['Water_Usage_Liters'].corr(promedio_impacto_por_pais['Waste_Production_KG'], method='pearson')
print(f"Correlación entre Promedio de Uso de Agua y Promedio de Producción de Residuos por País: {correlacion_agua_residuos:.2f}")

#Gráfico de dispersión para visualizar la relación entre dos métricas
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Carbon_Footprint_MT', y='Water_Usage_Liters', data=promedio_impacto_por_pais, color="#33ff4f")
for i, row in promedio_impacto_por_pais.iterrows():
    plt.annotate(i, (row['Carbon_Footprint_MT'], row['Water_Usage_Liters']), textcoords="offset points", xytext=(0,10), ha='center')
plt.title('Correlación entre Promedio de Huella de Carbono y Promedio de Uso de Agua por País')
plt.xlabel('Promedio de Huella de Carbono (MT)')
plt.ylabel('Promedio de Uso de Agua (Litros)')
plt.grid(True)
plt.tight_layout()
plt.savefig("../images/carbono_water.jpg")

#Gráfico de dispersión para visualizar la relación entre dos métricas
plt.figure(figsize=(12, 8)) #Un poco más grande para dar espacio porque coincidian países en el mismo punto y no se podían leer
sns.scatterplot(x='Carbon_Footprint_MT', y='Waste_Production_KG', data=promedio_impacto_por_pais, color="#33ff4f", s=80) #Aumento un poco el tamaño de los puntos

for i, row in promedio_impacto_por_pais.iterrows():
    x_offset = 5  #Desplazamiento horizontal inicial
    y_offset = 5  #Desplazamiento vertical inicial

    #Ajustes específicos para evitar superposiciones observadas en la imagen
    if i == 'Japan':
        x_offset = -12
        y_offset = 5
    elif i == 'Italy':
        x_offset = -10
        y_offset = 5
    elif i == 'Brazil':
        x_offset = -15
        y_offset = 10
    elif i == 'Australia':
        x_offset = -18
        y_offset = 5
    elif i == 'India':
        x_offset = -12
        y_offset = 5
    elif i == 'China':
        x_offset = -15
        y_offset = 5
    elif i == 'UK':
        x_offset = -8
        y_offset = 5
    elif i == 'France':
        x_offset = -14
        y_offset = 5
    elif i == 'Germany':
        x_offset = -20
        


    plt.annotate(i, (row['Carbon_Footprint_MT'], row['Waste_Production_KG']),
                 textcoords="offset points",
                 xytext=(x_offset, y_offset),
                 ha='left', va='bottom', fontsize=9)

plt.title('Correlación entre Promedio de Huella de Carbono y Promedio de Producción de Residuos por País', fontsize=14)
plt.xlabel('Promedio de Huella de Carbono (MT)', fontsize=12)
plt.ylabel('Promedio de Producción de Residuos (Kg)', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("../images/carbono_waste.jpg")

#Gráfico de dispersión para visualizar la relación entre dos métricas
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Water_Usage_Liters', y='Waste_Production_KG', data=promedio_impacto_por_pais, color="#33ff4f")
for i, row in promedio_impacto_por_pais.iterrows():
    plt.annotate(i, (row['Water_Usage_Liters'], row['Waste_Production_KG']), textcoords="offset points", xytext=(0,10), ha='center')
plt.title('Correlación entre Promedio de Uso de Agua y Promedio de Producción de Residuos por País')
plt.xlabel('Promedio de Uso de Agua (Litros)')
plt.ylabel('Promedio de Producción de Residuos (Kg)')
plt.grid(True)
plt.tight_layout()
plt.savefig("../images/water_waste.jpg")

#Análisis de la Hipótesis 6

#HIPÓTESIS 7: "La 'Market_Trend' de los productos podría variar según el país."

#Agrupo los datos por país y analizo la distribución de 'Market_Trend':
tendencia_por_pais = df1.groupby('Country')['Market_Trend'].value_counts()

print("Distribución de la Tendencia del Mercado por País:")
print(tendencia_por_pais)

#Calculo los porcentajes de cada tendencia por país:
tendencia_por_pais_porcentaje = df1.groupby('Country')['Market_Trend'].value_counts(normalize=True).mul(100).unstack(fill_value=0)

print("\nPorcentaje de la Tendencia del Mercado por País:")
print(tendencia_por_pais_porcentaje)

#Visualizo la distribución de 'Market_Trend' por país:
#Gráfico de barras apiladas: Muestra la proporción de cada tendencia dentro de cada país como segmentos de una barra.
tendencia_por_pais_porcentaje.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Distribución de la Tendencia del Mercado por País')
plt.xlabel('País')
plt.ylabel('Porcentaje')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Tendencia del Mercado')
plt.tight_layout()
plt.savefig("../images/market_trend_por_pais.jpg")

#Gráfico de barras agrupadas:
tendencia_por_pais_porcentaje.plot(kind='bar', figsize=(12, 7))
plt.title('Tendencia del Mercado por País')
plt.xlabel('País')
plt.ylabel('Porcentaje')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Tendencia del Mercado')
plt.tight_layout()
plt.savefig("../images/market_trend_por_pais_agrupadas.jpg")

#El mismo estudio que el anterior pero también por Años
tendencia_por_pais_anio_porcentaje = df.groupby(['Country', 'Year'])['Market_Trend'].value_counts(normalize=True).mul(100).unstack(fill_value=0)

print("\nPorcentaje de la Tendencia del Mercado por País y Año:")
print(tendencia_por_pais_anio_porcentaje)

#Visualización de la distribución de 'Market_Trend' por país y año:
paises_unicos = df1['Country'].unique()
num_paises = len(paises_unicos)
num_anios = df1['Year'].nunique()
tendencias_unicas = df1['Market_Trend'].unique()

fig, axes = plt.subplots(nrows=num_paises, ncols=1, figsize=(12, 5 * num_paises))

for i, pais in enumerate(paises_unicos):
    datos_pais = tendencia_por_pais_anio_porcentaje.loc[pais]
    datos_pais.plot(kind='bar', ax=axes[i], legend=True)
    axes[i].set_title(f'Tendencia del Mercado en {pais} por Año')
    axes[i].set_ylabel('Porcentaje')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("../images/market_trend_por_pais_año.jpg")

#Análisis de la Hipótesis 7

#HIPÓTESIS 8: "El nivel de sostenibilidad, inferido de la 'Sustainability_Rating', varía significativamente entre los diferentes países.

#Cuento la frecuencia de cada 'Sustainability_Rating' por país
conteo_sostenibilidad_por_pais = df1.groupby('Country')['Sustainability_Rating'].value_counts().unstack(fill_value=0)

print("Distribución de la Calificación de Sostenibilidad por País (Conteo):\n", conteo_sostenibilidad_por_pais)

#Calculo el porcentaje de cada 'Sustainability_Rating' por país
porcentaje_sostenibilidad_por_pais = df1.groupby('Country')['Sustainability_Rating'].value_counts(normalize=True).mul(100).unstack(fill_value=0)

print("\nDistribución de la Calificación de Sostenibilidad por País (Porcentaje):\n", porcentaje_sostenibilidad_por_pais)

#Visualización 1: Gráfico de barras apiladas (porcentaje)
porcentaje_sostenibilidad_por_pais.plot(kind='bar', stacked=True, figsize=(12, 7))
plt.title('Distribución Porcentual de la Calificación de Sostenibilidad por País')
plt.xlabel('País')
plt.ylabel('Porcentaje de Marcas')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Calificación de Sostenibilidad')
plt.tight_layout()
plt.savefig("../images/sostenibilidad_pais.jpg")

#Visualización 2: Gráfico de barras agrupadas (conteo)
conteo_sostenibilidad_por_pais.plot(kind='bar', figsize=(12, 7))
plt.title('Distribución de la Calificación de Sostenibilidad por País (Conteo)')
plt.xlabel('País')
plt.ylabel('Número de Marcas')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Calificación de Sostenibilidad')
plt.tight_layout()
plt.savefig("../images/sostenibilidad_pais_agrupadas.jpg")

#Visualización 3: Box plots de una métrica numérica asociada a la sostenibilidad (si existe)
#Huella de Carbono
plt.figure(figsize=(12, 7))
sns.boxplot(x='Country', y='Carbon_Footprint_MT', hue='Sustainability_Rating', data=df1, color='#d2d5d3')
plt.title('Huella de Carbono por País y Calificación de Sostenibilidad')
plt.xlabel('País')
plt.ylabel('Huella de Carbono (MT)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.legend(title='Sustainability_Rating', loc='upper right')
plt.savefig("../images/sostenibilidad_pais_carbon.jpg")

#Uso de Agua
plt.figure(figsize=(12, 7))
sns.boxplot(x='Country', y="Water_Usage_Liters", hue='Sustainability_Rating', data=df1, color= "#78f5d3")
plt.title('Uso de agua por País y Calificación de Sostenibilidad')
plt.xlabel('País')
plt.ylabel('Uso de Agua (Litros)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.legend(title='Sustainability_Rating', loc='upper right')
plt.savefig("../images/sostenibilidad_pais_carbon.jpg")

#Producción de Residuos
plt.figure(figsize=(12, 7))
sns.boxplot(x='Country', y="Waste_Production_KG", hue='Sustainability_Rating', data=df, color= "#c87607")
plt.title('Producción de Residuos y Calificación de Sostenibilidad')
plt.xlabel('País')
plt.ylabel('Producción de residuos (Litros)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.legend(title='Sustainability_Rating', loc='upper right')
plt.savefig("../images/sostenibilidad_pais_residuos.jpg")

#Visualización 4: Gráfico de barras del porcentaje de la calificación más alta por país
#La calificación más alta es el número 1
calificacion_mas_alta = 1

marcas_alta_calificacion = df1[df1['Sustainability_Rating'] == calificacion_mas_alta].groupby('Country').size()
total_marcas_por_pais = df1['Country'].value_counts()
porcentaje_alta_calificacion = (marcas_alta_calificacion / total_marcas_por_pais * 100).fillna(0).sort_values(ascending=False)

plt.figure(figsize=(12, 7))
porcentaje_alta_calificacion.plot(kind='bar', color="#33ff4f")
plt.title(f'Porcentaje de Marcas con Calificación de Sostenibilidad {calificacion_mas_alta} por País')
plt.xlabel('País')
plt.ylabel('Porcentaje de Marcas')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"../images/porcentaje_calificacion_mas_alta.jpg")

#Visualización 5: Gráfico de barras del porcentaje de la calificación más baja por país
#La calificación más baja es 4
calificacion_mas_baja = 4

marcas_alta_calificacion = df1[df1['Sustainability_Rating'] == calificacion_mas_baja].groupby('Country').size()
total_marcas_por_pais = df1['Country'].value_counts()
porcentaje_alta_calificacion = (marcas_alta_calificacion / total_marcas_por_pais * 100).fillna(0).sort_values(ascending=False)

plt.figure(figsize=(12, 7))
porcentaje_alta_calificacion.plot(kind='bar', color='red')  # Puedes cambiar el color si lo deseas
plt.title(f'Porcentaje de Marcas con Calificación de Sostenibilidad {calificacion_mas_baja} por País')
plt.xlabel('País')
plt.ylabel('Porcentaje de Marcas')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"../images/porcentaje_calificacion_mas_baja.jpg")

#Análisis de la Hipótesis 8

