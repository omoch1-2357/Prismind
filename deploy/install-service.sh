#!/bin/bash
#
# install-service.sh - Install and configure Prismind training service
#
# This script sets up the Prismind training service on a Linux system.
# Requirements Coverage: 11.7 (systemd service file for background training)
#
# Usage:
#   sudo ./install-service.sh [options]
#
# Options:
#   --install-dir DIR   Installation directory (default: /opt/prismind)
#   --user USER         Service user (default: prismind)
#   --no-user           Don't create service user
#   --uninstall         Remove service and user
#   --help              Show this help
#

set -e

# Default configuration
INSTALL_DIR="/opt/prismind"
SERVICE_USER="prismind"
CREATE_USER=true
UNINSTALL=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        --user)
            SERVICE_USER="$2"
            shift 2
            ;;
        --no-user)
            CREATE_USER=false
            shift
            ;;
        --uninstall)
            UNINSTALL=true
            shift
            ;;
        --help)
            echo "Usage: sudo $0 [options]"
            echo ""
            echo "Options:"
            echo "  --install-dir DIR   Installation directory (default: /opt/prismind)"
            echo "  --user USER         Service user (default: prismind)"
            echo "  --no-user           Don't create service user"
            echo "  --uninstall         Remove service and user"
            echo "  --help              Show this help"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check root
if [[ $EUID -ne 0 ]]; then
    print_error "This script must be run as root (use sudo)"
    exit 1
fi

# Uninstall
if $UNINSTALL; then
    echo "Uninstalling Prismind training service..."

    # Stop and disable service
    systemctl stop prismind-training 2>/dev/null || true
    systemctl disable prismind-training 2>/dev/null || true

    # Remove service file
    rm -f /etc/systemd/system/prismind-training.service
    systemctl daemon-reload

    # Remove user (optional)
    read -p "Remove service user '$SERVICE_USER'? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        userdel "$SERVICE_USER" 2>/dev/null || true
        print_success "User removed"
    fi

    print_success "Service uninstalled"
    exit 0
fi

echo "Installing Prismind training service..."
echo "  Install directory: $INSTALL_DIR"
echo "  Service user: $SERVICE_USER"
echo ""

# Create service user
if $CREATE_USER; then
    if ! id "$SERVICE_USER" &>/dev/null; then
        echo "Creating service user '$SERVICE_USER'..."
        useradd --system --no-create-home --shell /bin/false "$SERVICE_USER"
        print_success "User created"
    else
        print_warning "User '$SERVICE_USER' already exists"
    fi
fi

# Create directories
echo "Creating directories..."
mkdir -p "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR/checkpoints"
mkdir -p "$INSTALL_DIR/logs"
mkdir -p "$INSTALL_DIR/python"
print_success "Directories created"

# Set ownership
echo "Setting ownership..."
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
chmod 755 "$INSTALL_DIR"
chmod 755 "$INSTALL_DIR/checkpoints"
chmod 755 "$INSTALL_DIR/logs"
print_success "Ownership set"

# Copy service file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="$SCRIPT_DIR/prismind-training.service"

if [[ ! -f "$SERVICE_FILE" ]]; then
    print_error "Service file not found: $SERVICE_FILE"
    exit 1
fi

echo "Installing service file..."

# Update paths in service file
sed -e "s|/opt/prismind|$INSTALL_DIR|g" \
    -e "s|User=prismind|User=$SERVICE_USER|g" \
    -e "s|Group=prismind|Group=$SERVICE_USER|g" \
    "$SERVICE_FILE" > /etc/systemd/system/prismind-training.service

print_success "Service file installed"

# Reload systemd
echo "Reloading systemd..."
systemctl daemon-reload
print_success "Systemd reloaded"

echo ""
echo "========================================"
echo "Installation complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Copy your Prismind files to $INSTALL_DIR:"
echo "   sudo cp -r python/* $INSTALL_DIR/python/"
echo "   sudo cp patterns.csv $INSTALL_DIR/"
echo ""
echo "2. Install Python dependencies (if using venv):"
echo "   sudo -u $SERVICE_USER python3 -m venv $INSTALL_DIR/venv"
echo "   sudo -u $SERVICE_USER $INSTALL_DIR/venv/bin/pip install -e ."
echo ""
echo "3. Set correct ownership:"
echo "   sudo chown -R $SERVICE_USER:$SERVICE_USER $INSTALL_DIR"
echo ""
echo "4. Enable and start the service:"
echo "   sudo systemctl enable prismind-training"
echo "   sudo systemctl start prismind-training"
echo ""
echo "5. Check status:"
echo "   sudo systemctl status prismind-training"
echo "   journalctl -u prismind-training -f"
echo ""
