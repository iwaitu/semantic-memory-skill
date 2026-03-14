#!/bin/bash
# Manage gRPC Semantic Memory Service

SERVICE_NAME="com.semantic-memory.grpc-server"
PLIST_PATH="$HOME/Library/LaunchAgents/${SERVICE_NAME}.plist"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

case "$1" in
    start)
        if [ ! -f "$PLIST_PATH" ]; then
            echo "❌ Service not installed. Run: $0 install"
            exit 1
        fi
        launchctl load "$PLIST_PATH" 2>/dev/null
        echo "✅ Service started"
        ;;
    stop)
        launchctl unload "$PLIST_PATH" 2>/dev/null
        echo "✅ Service stopped"
        ;;
    restart)
        launchctl unload "$PLIST_PATH" 2>/dev/null
        sleep 1
        launchctl load "$PLIST_PATH" 2>/dev/null
        echo "✅ Service restarted"
        ;;
    status)
        if launchctl list | grep -q "$SERVICE_NAME"; then
            echo "🟢 Service is running"
            launchctl list | grep "$SERVICE_NAME"
        else
            echo "🔴 Service is not running"
        fi
        ;;
    install)
        # Copy plist to LaunchAgents
        cp "$PROJECT_DIR/config/${SERVICE_NAME}.plist" "$PLIST_PATH"
        chmod 644 "$PLIST_PATH"
        echo "✅ Service installed"
        # Start the service
        launchctl load "$PLIST_PATH" 2>/dev/null
        echo "✅ Service started"
        ;;
    uninstall)
        launchctl unload "$PLIST_PATH" 2>/dev/null
        rm -f "$PLIST_PATH"
        echo "✅ Service uninstalled"
        ;;
    logs)
        tail -f "$PROJECT_DIR/logs/grpc-server.out"
        ;;
    errors)
        tail -f "$PROJECT_DIR/logs/grpc-server.err"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|install|uninstall|logs|errors}"
        exit 1
        ;;
esac
