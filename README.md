# Fine-Tuning do Falcon 7B com LoRA

Este repositório contem um notebook completo para realizar fine-tuning do modelo Falcon 7B utilizando LoRA (Low-Rank Adaptation). O objetivo é permitir a personalização desse LLM (Large Language Model) sem precisar modificar todos os seus pesos, tornando o processo mais eficiente e acessível.

## Contextualização

Fine-Tuning é uma técnica utilizada para ajustar um modelo de IA pré-treinado em um novo conjunto de dados, permitindo que ele aprenda novas informações sem perder o conhecimento prévio.

**Por que isso é importante?**
Os modelos de linguagem grandes (LLMs) são treinados em bilhões de textos, mas nem sempre conhecem informações específicas. Com fine-tuning, podemos adaptar um modelo a domínios específicos, como atendimento ao cliente, medicina, direito ou engenharia.

**Este projeto permite:**

- Ajustar o Falcon 7B para responder perguntas personalizadas.
- Utilizar LoRA para reduzir o consumo de memória.
- Criar uma API para servir o modelo após o treinamento.

## O que é o Falcon 7B?

O Falcon 7B é um LLM (Large Language Model) desenvolvido pelo Technology Innovation Institute (TII). Ele faz parte da família de modelos Falcon e foi treinado em textos de alta qualidade para gerar respostas precisas e coerentes.

**Características do Falcon 7B:**

- Possui 7 bilhões de parâmetros.
- Modelo causal (Causal Language Model - CLM), usado para prever a próxima palavra.
- Open-source, permitindo personalizações.
- Suporta fine-tuning com LoRA, tornando o treinamento mais eficiente.

**Por que escolhi o Falcon 7B?**

- Ele oferece um ótimo equilíbrio entre desempenho e custo computacional.
- Pode ser treinado em GPUs acessíveis (como A100 e RTX 3090).
- Open-source, permitindo modificações e personalizações.

## Quando fazer fine-tuning?

Nem sempre precisamos treinar um LLM do zero. O fine-tuning é útil quando queremos especializar o modelo em um domínio ou tarefa específica.

**Quando fazer Fine-Tuning?**

- Quando o modelo base não entende um domínio específico (Ex: medicina, direito, engenharia).
- Quando precisamos de um chatbot personalizado (Ex: um assistente para suporte técnico).
- Quando queremos que o modelo siga um tom de linguagem específico (Ex: respostas mais formais ou descontraídas).
- Quando queremos adicionar novos conhecimentos sem treinar um modelo do zero (Ex: modelos de IA jurídicos).

**Vantagens do Fine-Tuning:**

- Evita ter que treinar um modelo do zero, economizando tempo e custo computacional.
- Melhora a precisão em tarefas específicas, como atendimento ao cliente ou geração de código.
- Funciona mesmo em GPUs menores usando técnicas como LoRA.

💡 Se o modelo base já responde bem às suas necessidades, você pode apenas usá-lo diretamente sem fazer fine-tuning.

## Estrutura do Repositório

| **Diretório/Arquivo** | **Descrição**                               |
| --------------------- | ------------------------------------------- |
| 📜 `README.md`        | Documentação do projeto                     |
| 📂 `notebooks/`       | Contém o notebook de treinamento            |
| 📂 `models/`          | Diretório onde o modelo treinado será salvo |
| 📂 `api/`             | Código da API para servir o modelo          |

## Como usar este projeto?

### Por que você precisa de uma GPU para o treinamento?

Os modelos de linguagem grandes (LLMs), como o Falcon 7B, possuem bilhões de parâmetros e exigem múltiplos cálculos de matriz durante o treinamento. Isso torna imprescindível o uso de uma GPU (Unidade de Processamento Gráfico), que pode acelerar os cálculos em comparação com uma CPU.

**Benefícios do uso de GPU no treinamento:**

- Processamento muito mais rápido do que CPUs para cálculos matriciais.
- Melhor aproveitamento da memória VRAM, permitindo treinar grandes modelos.
- Compatibilidade com bibliotecas otimizadas como PyTorch + CUDA.

**Usando o Google Colab Pro para o treinamento**
Se você não tem uma GPU local potente, o Google Colab Pro é uma ótima opção para treinar modelos de IA. Ele oferece acesso a GPUs poderosas, como NVIDIA A100 e V100, por um custo acessível.

**Vantagens do Google Colab Pro:**

- Permite treinar modelos sem precisar de uma GPU local.
- Acesso a GPUs NVIDIA A100 (40GB VRAM) para processar modelos grandes.
- Integração fácil com Google Drive para salvar checkpoints do modelo.

📌 Este notebook foi desenvolvido e testado no Google Colab Pro, utilizando uma GPU NVIDIA A100.

### Treinamento do modelo

O notebook inclui os passos para fazer fine-tuning do Falcon 7B com LoRA.
Basta executar as células no Jupyter Notebook ou no Google Colab Pro.

📌 Dica: No Google Colab, ative o uso de GPU A100 acessando:

`Runtime > Change runtime type > Hardware accelerator > GPU.`

#### Realizando inferências

Após o treinamento, você pode disponibilizar o modelo por meio de uma API REST que está no diretório `api` que foi construída utilizando FastAPI. Para isso, faça o download do modelo, que você acobou de treinar no Colab e salve-o no diretório `model` desse repositório.

#### Instale as dependências:

```bash
pip install transformers datasets accelerate peft bitsandbytes fastapi uvicorn

```

#### Execute o seguinte comando para iniciar o servidor:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Agora, sua API estará disponível e você pode visualizar o Swagger no endereço abaixo:

```
http://localhost:8000/docs
```

#### Teste via `curl` no Terminal

Teste a API diretamente pelo terminal:

```bash
curl -X 'POST' \
  'http://localhost:8000/generate/' \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "O futuro da inteligência artificial é", "max_length": 50, "temperature": 0.7}'
```

## Contribuição e Melhorias

- Se quiser contribuir com melhorias, fique à vontade para enviar um Pull Request ou abrir uma Issue.
- Se tiver dúvidas, entre em contato!
