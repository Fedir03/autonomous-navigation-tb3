#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   ./run_navigation_stack.sh        # no compila, solo lanza las 3 terminales
#   ./run_navigation_stack.sh 1      # compila antes de lanzar

COMPILE_FLAG="${1:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-$SCRIPT_DIR}"

if [[ ! -d "$WORKSPACE_DIR/src" ]]; then
	echo "[ERROR] No se encontro carpeta src en: $WORKSPACE_DIR"
	echo "        Exporta WORKSPACE_DIR o mueve el script a la raiz del workspace."
	exit 1
fi

if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
	echo "[ERROR] No existe /opt/ros/jazzy/setup.bash"
	exit 1
fi

setup_ros() {
	if command -v sb >/dev/null 2>&1; then
		# Si tienes alias/comando sb en tu entorno, lo usa.
		sb
	else
		source /opt/ros/jazzy/setup.bash
	fi
}

if [[ "$COMPILE_FLAG" == "1" ]]; then
	echo "[1/4] Compilando autonomous_navigation..."
	cd "$WORKSPACE_DIR"
	setup_ros
	colcon build --packages-select autonomous_navigation
else
	echo "[1/4] Compilacion omitida (pasar '1' para compilar)."
fi

if [[ ! -f "$WORKSPACE_DIR/install/setup.bash" ]]; then
	echo "[ERROR] No existe $WORKSPACE_DIR/install/setup.bash"
	echo "        Ejecuta primero: colcon build --packages-select autonomous_navigation"
	exit 1
fi

detect_terminal_backend() {
	if command -v gnome-terminal >/dev/null 2>&1; then
		echo "gnome-terminal"
	elif command -v x-terminal-emulator >/dev/null 2>&1; then
		echo "x-terminal-emulator"
	elif command -v xfce4-terminal >/dev/null 2>&1; then
		echo "xfce4-terminal"
	elif command -v konsole >/dev/null 2>&1; then
		echo "konsole"
	elif command -v xterm >/dev/null 2>&1; then
		echo "xterm"
	else
		echo "none"
	fi
}

TERMINAL_BACKEND="$(detect_terminal_backend)"
if [[ "$TERMINAL_BACKEND" == "none" ]]; then
	echo "[ERROR] No se encontro emulador de terminal (gnome-terminal/x-terminal-emulator/xfce4-terminal/konsole/xterm)."
	exit 1
fi

open_terminal() {
	local title="$1"
	local command="$2"

	case "$TERMINAL_BACKEND" in
		gnome-terminal)
			gnome-terminal --title="$title" -- bash -ic "$command; exec bash" &
			;;
		x-terminal-emulator)
			x-terminal-emulator -T "$title" -e bash -ic "$command; exec bash" &
			;;
		xfce4-terminal)
			xfce4-terminal --title="$title" --hold -e "bash -ic '$command; exec bash'" &
			;;
		konsole)
			konsole --new-tab -p tabtitle="$title" -e bash -ic "$command; exec bash" &
			;;
		xterm)
			xterm -T "$title" -hold -e "bash -ic '$command; exec bash'" &
			;;
	esac
}

COMMON_SETUP="cd \"$WORKSPACE_DIR\" && (command -v sb >/dev/null 2>&1 && sb || source /opt/ros/jazzy/setup.bash) && source install/setup.bash"

CMD_BASE_MAP="$COMMON_SETUP && pkill -f 'base_map_server|lifecycle_manager_base_map' || true && ros2 launch autonomous_navigation load_base_map.launch.py"
CMD_CARTOGRAPHER="$COMMON_SETUP && ros2 launch turtlebot3_cartographer cartographer.launch.py"
CMD_NAV="$COMMON_SETUP && ros2 run autonomous_navigation autonomous_navigation"

echo "[2/4] Abriendo terminal 1: base_map"
open_terminal "1_base_map" "$CMD_BASE_MAP"

sleep "${START_DELAY_BASE_MAP:-3}"

echo "[3/4] Abriendo terminal 2: cartographer"
open_terminal "2_cartographer" "$CMD_CARTOGRAPHER"

sleep "${START_DELAY_CARTOGRAPHER:-5}"

echo "[4/4] Abriendo terminal 3: autonomous_navigation"
open_terminal "3_autonomous_navigation" "$CMD_NAV"

echo "Listo. Se lanzaron las 3 terminales en orden."
