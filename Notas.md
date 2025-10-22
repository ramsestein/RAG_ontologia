# Notas de mi Procedimiento

**AUTOR:** Oriol Farrés

## Día 1: 20 Octubre 2025 (8h)

Primera toma de contacto. Explicaciones, cursos Anthropic, huella...


---


## Día 2: 21 Octubre 2025 (8h)

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

Antes de tratar estrategia 4, solucionaré SNOBERT.


---


## Día 3: 22 Octubre 2025 (4h)
Voy a empezar tratando de descargarme el modelo adecuado de BERT con Hugging Face.
Parece que sí estoy descargando el modelo real... 
Antes de seguir voy a hacer una reorganización de programas:
    - Cambio nombres de estrategias a 01, 02, 03, 04.
    - Cambiar nombre de la comparativa de todos a all_evaluate_strategies.py.
    - Crear evaluate_strategies.py, que se ejecuta con el argumento -<strategyID> para escojer que algoritmo ejecutar.
    - Cambio el nombre del directorio real_strategies a strategies.
Después de todo esto me doy cuenta que puedo hacer una implementación mucho más modular:
    Puedo tener solo un fichero evaluate_strategies.py y si le entra strategyID = 0 o simplemente si no se le asigna ningún parámetro (0 por defecto) que ejecute la comparativa general, que simplemente debe ser un for con un vector de las 3 estrategias válidas (01, 02 y 04) + una comparativa final de la métricas F1, recall y precision.
    Además, en lugar de estar hard-codeado en el mismo fichero, puede llamar a una clase Metrics con funciones como compare_N_metrics (N as argument) donde N es el numero de estrategias con las que compararé, siempre 1, menos en el caso de strategyID = 0, que ahí comparará con N = 3. Por ejemplo ora que sea print_metrics...

Vale, re-estructuración hecha. Hago push.

Antes de ponerme a mejorar el gpt 4o, que ya tengo una idea: (primero cachear los 14k embeddings, para solo tener que tardar la 1a vez), voy a:
    1. SNOBERT: Arreglar warnings, asegurarme que está loadeando correctamente el modelo-> luego arreglar tema corte de palabras.
