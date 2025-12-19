#!/usr/bin/env bash
set -e

echo "== 📊 AlphaTech Analyzer – Commit Asistido =="

echo "Tipos disponibles:"
echo "  feat  - nueva funcionalidad (ej. nueva técnica estadística)"
echo "  fix   - corrección de error en código o limpieza"
echo "  docs  - documentación (README, reportes, diapositivas)"
echo "  data  - actualización de datasets o scripts de descarga"
echo "  chore - mantenimiento (carpetas, requerimientos, gitignore)"

read -p "Tipo de commit (feat/fix/docs/data/chore): " type
read -p "Resumen breve (ej: configurar entorno base): " summary
read -p "Número de issue (solo número, ej: 1): " issue

# Limpieza de espacios
type="$(echo "$type" | tr '[:upper:]' '[:lower:]' | xargs)"
summary="$(echo "$summary" | xargs)"
issue="$(echo "$issue" | xargs)"

if [[ -z "$type" || -z "$summary" || -z "$issue" ]]; then
    echo "Error: tipo, resumen e issue son obligatorios."
    exit 1
fi

case "$type" in
    feat|fix|docs|data|chore)
        ;;
    *)
        echo "Error: tipo '$type' no válido."
        exit 1
        ;;
esac

# Formato: tipo: resumen (#número)
message="$type: $summary (#$issue)"

echo
echo "Mensaje de commit generado:"
echo "  $message"
echo

read -p "¿Confirmar y ejecutar 'git commit'? [s/N]: " confirm

if [[ "$confirm" == "s" || "$confirm" == "S" ]]; then
    git commit -m "$message"
else
    echo "Commit cancelado."
fi
