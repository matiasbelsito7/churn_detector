---
name: submit-change
description: Verificar, commitear y publicar un cambio de una tarea (T-NN) del proyecto: black, ruff, mypy, pytest, pre-commit, revisión del diff, commit único y push, y validación de los workflows de CI/CD. Usar cuando una tarea de docs/tasks.md esté implementada y haya que entregarla en git y confirmar que los workflows pasan.
---

# Skill: submit-change

Procedimiento para cerrar y publicar un cambio correspondiente a una tarea del
proyecto (`docs/tasks.md`, identificada como `T-NN`). Sustituye al criterio
personal: ante la duda, prevalece la rutina de cambio de `AGENTS.md` (sección 4).

## Cuándo usar

- Acabas de implementar una tarea `T-NN` y quieres entregarla (verificar,
  commitear, pushear) y confirmar que CI/CD pasa.
- Un flujo de verificación falló y hay que corregir, recomitear y re-pushear.

## Prerequisitos

- `git status` sin cambios pendientes de otras tareas. Solo se publica un cambio
  por tarea (no mezclar capas).
- La herramienta de la fase correspondiente existe (T-02 configura lint,
  typecheck, tests y pre-commit; T-21 configura CI/CD). Si una herramienta o
  workflow todavía no está configurada, se anota el pendiente y se continúa;
  **no se inventa** tal herramienta ni se cubren errores fingiendo resultados.

## 1. Verificación local

Ejecutar en orden y corregir hasta dejar todo en verde. No se avanza al
siguiente paso con errores de una herramienta anterior.

1. `black --check src tests`
2. `ruff check src tests`
3. `mypy src`
4. `pytest`
5. `pre-commit run --all-files`

Si aparecen errores (formato/lint/tipos/tests), corregir en la capa
correspondiente y repetir el ciclo hasta que todo pase.

## 2. Revisión del diff

- `git status`
- `git diff --stat`
- `git diff` para revisar el contenido exacto.

Criterios:
- Un cambio por tarea: no mezclar tareas ni capas distintas (AGENTS §4).
- Sin secretos ni credenciales (AGENTS §5).
- Funcionalidad nueva acompañada de sus pruebas (AGENTS §5).
- No tocar documentos de gobierno (`docs/constitution.md`, `docs/specs.md`,
  `docs/tasks.md`) sin autorización explícita.

## 3. Commit

- Mensaje en español, imperativo y con referencia a la tarea, p. ej.:
  `T-02: configurar entorno reproducible con lint, typecheck y tests`.
- `git add` solo de los archivos de la tarea.
- `git commit -m "..."`.

## 4. Push

- Verificar que el remoto existe (`git remote -v`).
- `git push`
- Si el push es rechazado, `git pull --rebase` y volver a intentar (resolver
  conflictos sin mezclar capas).

## 5. Chequeo de CI/CD

Si existen workflows en `.github/workflows`:

1. `gh run list` para identificar el run del push.
2. `gh run watch <run-id> --exit-status` y esperar resultado.
3. Si falla: investigar el log, corregir, commitear (mismo formato) y re-pushear.

Si todavía no existen workflows (hasta T-21): registrar como pendiente en la
tarea y no bloquear la declaración de completitud local.

## 6. Monitoreo en segundo plano (opcional)

Después del push, para no bloquear el siguiente trabajo:

1. Lanzar un subagente con la herramienta `task` (background) con la
   instrucción: vigilar el run de CI más reciente con `gh run watch <run-id>
   --exit-status` y reportar el resultado y cualquier fallo con su causa.
2. Continuar con la siguiente tarea mientras el subagente vigila.
3. Ante el aviso de fallo, priorizar la corrección antes de avanzar más.

## Recordatorios

- Verificación realizada no es completitud si el push/CI quedan pendientes:
  declarar siempre el estado real de cada paso.
- Ante un bloqueo (datos faltantes, ambigüedad, dependencia no resuelta),
  exponerlo (AGENTS §3) en lugar de improvisar.
