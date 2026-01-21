# Guía de Contribución

¡Gracias por su interés en contribuir al proyecto **RAG Ontología**!

Este documento establece las pautas para contribuir al código base. Seguir estas directrices ayuda a mantener la calidad del código, facilita la colaboración y asegura que el proyecto sea sostenible a largo plazo.

## 🤝 Cómo Contribuir

Valoramos las contribuciones de todos los miembros del equipo y colaboradores autorizados. Aquí hay algunas formas en las que puede ayudar:

1.  **Reportar Errores (Bugs):** Si encuentra un error, por favor cree un "Issue" detallado describiendo el problema, los pasos para reproducirlo y el comportamiento esperado.
2.  **Sugerir Mejoras:** Las ideas para nuevas funcionalidades o mejoras en el rendimiento son bienvenidas.
3.  **Documentación:** Mejorar la documentación existente o añadir nuevos ejemplos es tan valioso como escribir código.
4.  **Desarrollo de Código:** Solucionar problemas abiertos o implementar nuevas características.

## workflow de Desarrollo

Para mantener el historial limpio y ordenado, recomendamos el siguiente flujo de trabajo:

1.  **Fork y Clonar:** Si es externo al equipo principal, trabaje sobre un "Fork". Si es miembro del equipo, clone el repositorio directamente.
2.  **Ramas (Branches):** Cree una nueva rama para cada funcionalidad o corrección. Use nombres descriptivos:
    *   `feature/nueva-funcionalidad`
    *   `fix/correccion-error`
    *   `docs/actualizacion-readme`
3.  **Commits:** Haga commits pequeños y frecuentes con mensajes claros y descriptivos.
    *   *Malo:* "arreglos"
    *   *Bueno:* "Fix: corrige error de indexación en generador de grafos"
4.  **Pull Requests (PR):** Cuando su código esté listo, abra un PR hacia la rama `main`. Asegúrese de describir los cambios realizados y vincular cualquier "Issue" relacionado.

## 💎 Estándares de Código

Para mantener la consistencia en el proyecto:

*   **Estilo Python:** Seguimos la guía de estilo **PEP 8**. Use herramientas como `black` o `flake8` para formatear su código antes de enviar.
*   **Tipado:** Se recomienda encarecidamente el uso de **Type Hints** de Python para mejorar la legibilidad y detección de errores.
*   **Comentarios:** Documente las funciones complejas y las decisiones de diseño no obvias. Use Docstrings para clases y módulos.
*   **Estructura de Archivos:** Mantenga la separación lógica de componentes (ver estructura en `README.md`). No mezcle lógica de negocio con scripts de utilidad si es posible.

## 🧪 Pruebas

Asegúrese de que su código no rompa la funcionalidad existente.
*   Si añade una nueva funcionalidad, intente incluir pruebas unitarias o al menos un script de demostración que verifique su funcionamiento.
*   Ejecute el código localmente antes de enviar un PR.

## 📜 Código de Conducta

Esperamos que todos los colaboradores mantengan un ambiente profesional, respetuoso y colaborativo. Las críticas constructivas son bienvenidas; la falta de respeto no será tolerada.

---
¡Gracias por ayudar a avanzar la informática clínica!
