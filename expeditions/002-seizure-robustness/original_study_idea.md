Desenho do estudo (versão revista)

Título: Robustez de algoritmos de deteção automática de crises epiléticas a erros técnicos de aquisição do EEG

Enquadramento
Os algoritmos de deteção automática de crises perdem muito desempenho quando saem dos dados em que foram desenvolvidos. No maior desafio independente até hoje, o melhor algoritmo teve um F1 de 32%, e os autores apontam os artefactos dos dados clínicos reais como um dos fatores. Ainda não se sabe quanto do problema se deve especificamente a erros técnicos de aquisição, que dependem do trabalho do TSDT.

Pergunta de investigação
Qual é o impacto dos erros técnicos de aquisição no desempenho de algoritmos pré-treinados de deteção de crises, e esse impacto difere entre métodos clássicos e de deep learning?

Hipóteses

	•	H1: os erros técnicos reduzem o desempenho de forma dependente do tipo e da intensidade do erro.
	•	H2: o algoritmo de deep learning e o clássico diferem na sua robustez.

Tipo de estudo
Estudo experimental, quantitativo, com dados secundários públicos e desenho emparelhado: cada registo funciona como o seu próprio controlo.

Dados

	•	Siena Scalp EEG Database (PhysioNet): adultos, montagem 10-20 referencial, compatível com o SzCORE.
	•	Convertidos para o formato SzCORE com o código oficial.
	•	Todos os doentes com pelo menos uma crise anotada.

Algoritmos (pré-treinados, sem treino adicional e sem a Siena nos dados de treino)

	1.	eventNet: deep learning (U-Net e CNN), treinado só com TUH.
	2.	Gotman: detetor clássico, sem treino.

Intervenção: erros técnicos simulados
Os erros são aplicados ao nível do elétrodo, antes do re-referenciamento. Isto é tecnicamente importante: os algoritmos usam montagem de média comum, por isso um único elétrodo mau contamina todos os canais. É um ponto forte a explorar na apresentação.

Erro	Simulação	Níveis
Interferência da rede	sinusoide de 50 Hz	3 amplitudes
Impedância elevada	ruído de banda larga	3 SNR
Elétrodo desligado	sinal plano ou deriva	1, 2 ou 3 elétrodos
Artefacto muscular	ruído de alta frequência em rajadas	3 intensidades
Deslocamento de elétrodo	mistura com o elétrodo vizinho	3 proporções

Variáveis de resultado (métricas por evento do SzCORE, com a biblioteca oficial)
