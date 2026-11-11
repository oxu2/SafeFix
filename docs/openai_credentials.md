# API credentials

`utils.gpt.ChatBot` accepts an explicit `api_key` argument or reads
`OPENAI_API_KEY` from the process environment. The example notebooks read the
same environment variable. Configure it in the shell or a local secret manager;
never save a real value in source, notebook cells, outputs, or version control.
The code raises a clear error when neither source supplies a key.

The legacy OpenAI SDK calls in these examples are unchanged. A CLIProxy client
key authenticates to the configured proxy endpoint; it is distinct from an
OpenAI platform API key and must be used with the matching proxy endpoint and
supported model. Do not send a proxy client key to the default OpenAI endpoint.

## November 11, 2026 remediation

Removed hardcoded credentials from `utils/gpt.py`, `utils/run.ipynb`, and
`utils/celebA_use_case.ipynb`; cleared notebook outputs and execution counts.
A scan of tracked current files found no remaining OpenAI-format key literals.
The remote `main` tree also contained no matching key literals at this check.

Historical commits still contain the leaked key. Removing it from current files
does not revoke it or remove old Git objects. The key owner must revoke it in
the OpenAI account and inspect usage. GitHub alert #1 remains open until
revocation is confirmed. No history rewrite or force-push was performed.
