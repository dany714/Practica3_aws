"""Práctica 3: detección IQR, corrección de edad e imputación KNN.

Dataset y KNN k=2 del código base. Comparación adicional con k=3, k=5,
media y mediana sobre los mismos datos preparados. No requiere AWS.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer

BASE = Path(__file__).resolve().parent
SALIDAS = BASE / 'salidas'


def crear_dataset():
    """Reproduce las diez filas del enunciado sin agregar variables al KNN."""
    return pd.DataFrame({
        'edad':[25,30,np.nan,45,22,120,35,28,np.nan,40],
        'ingreso_mensual':[15000,22000,18000,np.nan,12000,950000,25000,21000,19000,np.nan],
        'scoring_credito':[650,700,680,710,590,800,670,640,620,730],
    })


def preparar(original):
    """Calcula IQR sobre ingresos observados; sustituye anomalías por NaN."""
    q1=original.ingreso_mensual.quantile(.25)
    q3=original.ingreso_mensual.quantile(.75)
    iqr=q3-q1
    inferior=q1-1.5*iqr
    superior=q3+1.5*iqr
    outliers=original.ingreso_mensual.lt(inferior)|original.ingreso_mensual.gt(superior)
    edad_erronea=original.edad.gt(100)
    preparado=original.copy()
    preparado.loc[outliers,'ingreso_mensual']=np.nan
    preparado.loc[edad_erronea,'edad']=np.nan
    return preparado,{'Q1':q1,'Q3':q3,'IQR':iqr,'limite_inferior':inferior,'limite_superior':superior},outliers,edad_erronea


def imputar(preparado):
    """Imputa cada alternativa de forma independiente, sin reutilizar resultados."""
    resultados={}
    for k in (2,3,5):
        array=KNNImputer(n_neighbors=k,weights='uniform',metric='nan_euclidean').fit_transform(preparado)
        resultados[f'KNN_k{k}']=pd.DataFrame(array,columns=preparado.columns,index=preparado.index)
    for estrategia in ('mean','median'):
        array=SimpleImputer(strategy=estrategia).fit_transform(preparado)
        resultados['Media' if estrategia=='mean' else 'Mediana']=pd.DataFrame(array,columns=preparado.columns,index=preparado.index)
    return resultados


def comparar(original,preparado,resultados):
    """Muestra únicamente las celdas que necesitan imputación."""
    filas=[]
    for i,col in zip(*np.where(preparado.isna().to_numpy())):
        nombre=preparado.columns[col]
        fila={'registro':int(i+1),'variable':nombre,'original':original.iloc[i,col]}
        for metodo,df in resultados.items():fila[metodo]=df.iloc[i,col]
        filas.append(fila)
    return pd.DataFrame(filas)


def validar(original,preparado,resultados):
    """Verifica conservación de filas y de todas las celdas observadas válidas."""
    observados=~preparado.isna().to_numpy()
    for nombre,limpio in resultados.items():
        if limpio.shape!=original.shape or limpio.isna().any().any():
            raise ValueError(f'{nombre}: estructura incorrecta o nulos restantes.')
        if not np.allclose(limpio.to_numpy()[observados],preparado.to_numpy()[observados]):
            raise ValueError(f'{nombre}: se modificaron valores observados válidos.')
        if not limpio.edad.between(0,100).all():
            raise ValueError(f'{nombre}: edad fuera del dominio.')


def tabla(df):
    vista=df.copy();vista.index=np.arange(1,len(vista)+1);vista.index.name='registro'
    return vista.to_string(float_format=lambda x:f'{x:.2f}')


def graficar(original,final):
    """Boxplots antes/después. Omite NaN solo para dibujar, no para limpiar."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for fila,(col,unidad) in enumerate([('edad','Años'),('ingreso_mensual','Ingreso mensual')]):
        for j,(nombre,df,color) in enumerate([('Original',original,'#ad4646'),('Después de limpieza y KNN k=2',final,'#197d70')]):
            ax=axes[fila,j]
            ax.boxplot(df[col].dropna(),patch_artist=True,widths=.4,
                       boxprops={'facecolor':color,'alpha':.35},
                       medianprops={'color':'#222222','linewidth':2},
                       flierprops={'marker':'o','markerfacecolor':color,'markeredgecolor':color})
            ax.set_xticks([1],[nombre],fontsize=10)
            ax.set_ylabel(unidad);ax.set_title('Edad' if col=='edad' else 'Ingreso mensual',fontsize=12)
            ax.grid(axis='y',alpha=.2)
            if col=='edad':ax.set_ylim(0,130)
            else:ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
            ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Distribución antes y después del tratamiento',fontsize=17,fontweight='bold')
    fig.supxlabel('Ingreso: cada panel usa su propia escala vertical para mostrar el detalle.',fontsize=10)
    path=SALIDAS/'boxplot_antes_despues.png'
    fig.savefig(path,dpi=180,facecolor='white');plt.close(fig)
    return path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paso',choices=['inicial','iqr','knn','comparar','graficos','todo'])
    args=parser.parse_args(argv)
    try:
        original=crear_dataset()
        preparado,limites,outliers,edad_erronea=preparar(original)
        resultados=imputar(preparado)
        comparacion=comparar(original,preparado,resultados)
        validar(original,preparado,resultados)
        SALIDAS.mkdir(parents=True,exist_ok=True)
        original.to_csv(SALIDAS/'datos_originales.csv',index=False,encoding='utf-8-sig')
        preparado.to_csv(SALIDAS/'datos_preparados.csv',index=False,encoding='utf-8-sig')
        for nombre,df in resultados.items():df.to_csv(SALIDAS/f'datos_{nombre.lower()}.csv',index=False,encoding='utf-8-sig')
        comparacion.to_csv(SALIDAS/'comparacion_imputaciones.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame([limites]).to_csv(SALIDAS/'limites_iqr.csv',index=False,encoding='utf-8-sig')
        print('PRÁCTICA 3 | LIMPIEZA DE DATOS AVANZADA')
        if args.paso in ('inicial','todo'):
            print('\nDATASET ORIGINAL | 10 registros x 3 columnas')
            print(tabla(original))
            print('\nFALTANTES POR COLUMNA');print(original.isna().sum().to_string())
            print('\nVALORES ANÓMALOS: registro 6, edad=120 e ingreso_mensual=950000.')
        if args.paso in ('iqr','todo'):
            print('\nDETECCIÓN IQR | ingreso_mensual')
            for nombre,valor in limites.items():print(f'{nombre}: {valor:,.2f}')
            print(f'Ingresos fuera de los límites: {int(outliers.sum())}')
            print(f'Edades mayores de 100: {int(edad_erronea.sum())}')
            print('\nDATASET PREPARADO | anomalías reemplazadas por NaN')
            print(tabla(preparado))
            print('\nFALTANTES A IMPUTAR');print(preparado.isna().sum().to_string())
        if args.paso in ('knn','todo'):
            print('\nDATASET FINAL | KNNImputer, k=2, pesos uniformes')
            print(tabla(resultados['KNN_k2']))
            print('\nFALTANTES FINALES');print(resultados['KNN_k2'].isna().sum().to_string())
            print('\nConservados los 10 registros y todos los valores observados válidos.')
        if args.paso in ('comparar','todo'):
            print('\nCOMPARACIÓN DE CELDAS IMPUTADAS')
            print(comparacion.to_string(index=False,float_format=lambda x:f'{x:.2f}'))
            print('\nCada método parte del mismo dataset preparado.')
            print('Media/mediana: calculadas sin el ingreso atípico y sin la edad errónea.')
            print('KNN conserva las escalas originales del código base; la escala influye en la distancia.')
            print('Diferencias entre métodos no demuestran mayor exactitud sin valores reales de referencia.')
        if args.paso in ('graficos','todo'):
            path=graficar(original,resultados['KNN_k2'])
            print('\nGRÁFICO GENERADO');print(path)
            print('Abrir salidas/boxplot_antes_despues.png en VS Code.')
        print('\nProceso completado correctamente.')
        return 0
    except (OSError,ValueError) as error:
        print(f'No se pudo completar la limpieza: {error}',file=sys.stderr)
        return 1


if __name__=='__main__':
    sys.exit(main())
