# Notas de mi Procedimiento

**AUTOR:** Oriol Farrés

## Día 1: 20 Octubre 2025

Primera toma de contacto. Explicaciones, cursos Anthropic, huella...


---


## Día 2: 21 Octubre 2025

He empezado instalando todas las herramientas para poder trabajar.
He hablado con ``Ramses`` para tener más idea de como plantear el proyecto.
Añado gitignore.
Voy a empezar corriendo el código por primera vez, hay problemas, voy a solucionarlos -> TODO list.
Voy a solucionar problemas 1 a 1, primero, voy a intentar installar Ollama. -> No puedo.
Voy a trabajar sin el ollama, así que simplemente comentaré la estrategia ollama.
Ahora arreglo los paths, más robusto->no warnings. Había errores con paths absolutos y relativos: 
    $ python benchmark/complete_real_comparison.py => Daba error.
Tengo problemas con el import de la strategia 4. Solucionado.
Solucionado warnings vsc.
Ahora estoy arreglando el formato de los outputs, creando carpetas para cada ejecucion. Solucionado.
Identificar porque los resultados de nuestra estrategia (4) son tan malos. Me ha comentado ``Ramses`` que me fije en la implementación general pero que empiece por tamaño de chunks, temperatura...
Básicamente hay un gran problema:
    Estrategia           F1-Score   Precision  Recall     Pred   Match  Tiempo
    1_KIRIs_REAL         0.8000     0.8381     0.7652     105    88     0.0s
    2_SNOBERT_REAL       0.3630     0.3072     0.4435     166    51     10.0s
    4_TU_RAG_GPT4o       0.0310     0.1429     0.0174     14     2      74.8s     

El recall es absurdamente malo. Primero toca entender qué es recall exactamente.
Según nuestro cálculo de métricas (Creo que podríamos añadir más en un futuro para mirar más cuáles son los mejores modelos), tenemos 2 variables:

1. `ground_truth`: Representa todas las anotaciones correctas que deberían ser encontradas.
-> **len(ground_truth)** representa el número total de positivos reales.
2. `exact_matches`: Cuenta cuántas de las predicciones de tu modelo coinciden exactamente.

$$
    recall = \dfrac{exact\_matches}{len(ground\_truth)}
$$

3. `len(predictions)`: Número total de predicciones que hizo el modelo, incluyendo tanto las predicciones correctas como las incorrectas (Falsos Positivos).

$$
    precision = \dfrac{exact\_matches}{len(predictions)}
$$

Es decir nuestro problema es que la precisión es baja, pero el recall es rídiculo. Voy a empezar con esto.

NOTAS 2a reunión del día con `Ramses`:
->El Snobert no funciona con el modelo que debería, instalarlo en local y ponerlo aquí, en el proyecto (hugging face). Así tendré los mismos resultados que los del README.md. Si ni con eso no es suficiente problema de cortes: e.g. "juancito" lo está dividiendo como "juan" "cito".

TEORÍA:
-BERT los coge casi todos, pero coge algunos que no debería.
-gpt coge algunos pero se deja muchos.
-> Él propone (cree) que la mejor ocion sera hacer un BERT y luego un pruning con un LLM.