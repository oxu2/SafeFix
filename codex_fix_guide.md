# Fixing the Codex Extension in Antigravity IDE

This document provides instructions to fix the Codex extension loading issue in Antigravity IDE on other computers.

## Problem Description

The Codex extension fails to load in Antigravity IDE due to two main issues:
1. **API Restrictions**: The extension requires proposed VS Code APIs (`chatSessionsProvider` and `languageModelProxy`), which are blocked by default unless explicitly allowed in the `product.json` configuration file of the IDE.
2. **Missing Binary**: Universal installations of the extension sometimes lack the required OS-specific native binary `bin/macos-aarch64/codex`.

---

## Resolution Steps

### Step 1: Allow Proposed APIs in product.json

1. Locate the application configuration file.
   - **On macOS**: `/Applications/Antigravity IDE.app/Contents/Resources/app/product.json`
   - **On Windows/Linux**: Locate the installation directory of Antigravity IDE and find the `resources/app/product.json` file.

2. Open `product.json` in a text editor.

3. Find the `"extensionEnabledApiProposals"` section. It will look like this:
   ```json
   "extensionEnabledApiProposals": {
       "ms-vscode.vscode-selfhost-test-provider": ["testObserver", "testRelatedCode", "attributableCoverage"],
       ...
   }
   ```

4. Add the entry for `"openai.chatgpt"` at the beginning of the block:
   ```json
   "extensionEnabledApiProposals": {
       "openai.chatgpt": ["languageModelProxy", "chatSessionsProvider"],
       "ms-vscode.vscode-selfhost-test-provider": ["testObserver", "testRelatedCode", "attributableCoverage"],
       ...
   }
   ```

5. Save the file.

### Step 2: Install the Platform-Specific Extension

If the Codex extension still fails to start or complains about a missing binary, you must ensure that you are running the platform-specific version:

1. Open the extension manager in Antigravity IDE.
2. If the extension is already installed, check if it is version `universal`.
3. If it is the universal version, uninstall it and install the platform-specific version matching the system architecture (for example, `darwin-arm64` for Apple Silicon Mac computers).

---

## Verification

After completing the steps above, restart Antigravity IDE. The Codex sidebar and commands will now open and function properly.
