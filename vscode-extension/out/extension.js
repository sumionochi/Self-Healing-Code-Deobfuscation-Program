"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.deactivate = exports.activate = void 0;
const vscode = require("vscode");
const path = require("path");
const fs = require("fs");
const node_fetch_1 = require("node-fetch");
function activate(context) {
    const disposable = vscode.commands.registerCommand('selfHealingCode.openPanel', () => {
        DeobfuscationPanel.createOrShow(context.extensionUri);
    });
    context.subscriptions.push(disposable);
}
exports.activate = activate;
function deactivate() {
    // No-op
}
exports.deactivate = deactivate;
class DeobfuscationPanel {
    static createOrShow(extensionUri) {
        const column = vscode.window.activeTextEditor
            ? vscode.window.activeTextEditor.viewColumn
            : undefined;
        // If we already have a panel, reveal it
        if (DeobfuscationPanel.currentPanel) {
            DeobfuscationPanel.currentPanel._panel.reveal(column);
            return;
        }
        // Otherwise, create a new panel
        const panel = vscode.window.createWebviewPanel('deobfuscationPanel', 'Self-Healing Code (File-Based)', column || vscode.ViewColumn.One, {
            enableScripts: true,
            localResourceRoots: [vscode.Uri.joinPath(extensionUri, 'media')],
            // Retain the webview’s context so it doesn't reset when hidden
            retainContextWhenHidden: true
        });
        DeobfuscationPanel.currentPanel = new DeobfuscationPanel(panel, extensionUri);
    }
    constructor(panel, extensionUri) {
        this._disposables = [];
        this._panel = panel;
        this._extensionUri = extensionUri;
        this._panel.webview.html = this._getHtmlForWebview();
        this._panel.onDidDispose(() => this.dispose(), null, this._disposables);
        // Listen for messages from the webview
        this._panel.webview.onDidReceiveMessage((msg) => __awaiter(this, void 0, void 0, function* () {
            if (msg.command === 'makeApiCall') {
                const { endpoint, method, payload } = msg.data;
                try {
                    const baseUrl = 'http://127.0.0.1:8000'; // or your server address
                    let response;
                    if (method === 'GET') {
                        // Possibly append query params
                        const qs = payload ? '?' + new URLSearchParams(payload).toString() : '';
                        response = yield (0, node_fetch_1.default)(baseUrl + endpoint + qs);
                    }
                    else {
                        response = yield (0, node_fetch_1.default)(baseUrl + endpoint, {
                            method,
                            headers: { 'Content-Type': 'application/json' },
                            body: payload ? JSON.stringify(payload) : undefined
                        });
                    }
                    if (!response.ok) {
                        const errorText = yield response.text();
                        this._panel.webview.postMessage({
                            command: 'apiResponse',
                            requestId: msg.requestId,
                            success: false,
                            error: errorText
                        });
                    }
                    else {
                        const data = yield response.json();
                        this._panel.webview.postMessage({
                            command: 'apiResponse',
                            requestId: msg.requestId,
                            success: true,
                            data
                        });
                    }
                }
                catch (error) {
                    this._panel.webview.postMessage({
                        command: 'apiResponse',
                        requestId: msg.requestId,
                        success: false,
                        error: error.toString()
                    });
                }
            }
        }), null, this._disposables);
    }
    _getHtmlForWebview() {
        const htmlPath = path.join(this._extensionUri.fsPath, 'media', 'panel.html');
        return fs.readFileSync(htmlPath, 'utf8');
    }
    dispose() {
        DeobfuscationPanel.currentPanel = undefined;
        while (this._disposables.length) {
            const item = this._disposables.pop();
            if (item) {
                item.dispose();
            }
        }
    }
}
//# sourceMappingURL=extension.js.map