import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import fetch from 'node-fetch';

export function activate(context: vscode.ExtensionContext) {
    const disposable = vscode.commands.registerCommand('selfHealingCode.openPanel', () => {
        DeobfuscationPanel.createOrShow(context.extensionUri);
    });
    context.subscriptions.push(disposable);
}

export function deactivate() {
    // No-op
}

class DeobfuscationPanel {
    public static currentPanel: DeobfuscationPanel | undefined;
    private readonly _panel: vscode.WebviewPanel;
    private readonly _extensionUri: vscode.Uri;
    private _disposables: vscode.Disposable[] = [];

    public static createOrShow(extensionUri: vscode.Uri) {
        const column = vscode.window.activeTextEditor
            ? vscode.window.activeTextEditor.viewColumn
            : undefined;

        // If we already have a panel, reveal it
        if (DeobfuscationPanel.currentPanel) {
            DeobfuscationPanel.currentPanel._panel.reveal(column);
            return;
        }

        // Otherwise, create a new panel
        const panel = vscode.window.createWebviewPanel(
            'deobfuscationPanel',
            'Self-Healing Code (File-Based)',
            column || vscode.ViewColumn.One,
            {
                enableScripts: true,
                localResourceRoots: [vscode.Uri.joinPath(extensionUri, 'media')],
                // Retain the webview’s context so it doesn't reset when hidden
                retainContextWhenHidden: true
            }
        );

        DeobfuscationPanel.currentPanel = new DeobfuscationPanel(panel, extensionUri);
    }

    private constructor(panel: vscode.WebviewPanel, extensionUri: vscode.Uri) {
        this._panel = panel;
        this._extensionUri = extensionUri;

        this._panel.webview.html = this._getHtmlForWebview();
        this._panel.onDidDispose(() => this.dispose(), null, this._disposables);

        // Listen for messages from the webview
        this._panel.webview.onDidReceiveMessage(
            async (msg) => {
                if (msg.command === 'makeApiCall') {
                    const { endpoint, method, payload } = msg.data;
                    try {
                        const baseUrl = 'http://127.0.0.1:8000'; // or your server address
                        let response;

                        if (method === 'GET') {
                            // Possibly append query params
                            const qs = payload ? '?' + new URLSearchParams(payload).toString() : '';
                            response = await fetch(baseUrl + endpoint + qs);
                        } else {
                            response = await fetch(baseUrl + endpoint, {
                                method,
                                headers: { 'Content-Type': 'application/json' },
                                body: payload ? JSON.stringify(payload) : undefined
                            });
                        }

                        if (!response.ok) {
                            const errorText = await response.text();
                            this._panel.webview.postMessage({
                                command: 'apiResponse',
                                requestId: msg.requestId,
                                success: false,
                                error: errorText
                            });
                        } else {
                            const data = await response.json();
                            this._panel.webview.postMessage({
                                command: 'apiResponse',
                                requestId: msg.requestId,
                                success: true,
                                data
                            });
                        }
                    } catch (error: any) {
                        this._panel.webview.postMessage({
                            command: 'apiResponse',
                            requestId: msg.requestId,
                            success: false,
                            error: error.toString()
                        });
                    }
                }
            },
            null,
            this._disposables
        );
    }

    private _getHtmlForWebview(): string {
        const htmlPath = path.join(this._extensionUri.fsPath, 'media', 'panel.html');
        return fs.readFileSync(htmlPath, 'utf8');
    }

    public dispose() {
        DeobfuscationPanel.currentPanel = undefined;
        while (this._disposables.length) {
            const item = this._disposables.pop();
            if (item) {
                item.dispose();
            }
        }
    }
}
