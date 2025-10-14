1. A 1–2 page PDF memo with the structure you feel best communicates your idea. We encour-
age, but do not require, the following sections:
• Motivation: Why is this problem worth solving? What would success of your method
enable that are difficult to do with other methods?
• Background / Related Work: Summarize key prior approaches and why they fall
short.
• Proposal: Describe your hypothesis, design, or direction of investigation.
• Evaluation: How would you test your proposal? What metrics or comparisons matter?
Are these different than current metrics in the literature?
Figures and diagrams are welcome where helpful. Please keep your proposal focused and
grounded in reality.
2. An implementation of your method accompanied by the evaluation(s) that best demonstrate
its effectiveness. You will use the Github repo provided over email.
3. A statement on how you used AI assistants to complete the assignments. We seek to under-
stand how you work with assistants and which parts of the assignment you are most strongly
responsible for.


Motivation: 

KV cache is important because it enables transformers to do inference in linear time instead of quadratic time. Re-computing the the keys and values of a token is very expensive because we have to run an attention that looks at all tokens before it. The problem with KV cache is that it takes up a lot of memory. GPUs not only have limited amount of high bandwidth memory, but also if our KV cache is too large, pulling the KV cache from HBM (high bandwidth memory) to SLM (shared local memory) and pushing the updates back can become a bottleneck. Indeed, transformers are often memory bottlenecked during token generation. Therefore, reducing the size of the KV cache enables speedy inference.  

As for why merged KV cache is important, I think there are two ways of looking at it. First it's practically useful. In addition to KV-cache quantization, it works along a difference axis to reduce the size of KV cache. When we quantize KV cache to 1-2 bits, I think we are already pushing the limits of how far we can go in this direction--we have reached information bottleneck. But working horizontally--merging kv cache is another way to further reduce memory. In fact I am aware that a lot of the SOTA quantization research has been shifting away from rounding-based quantization, towards bundling several dimensions of vectors in clever ways to reduce space. So, it's only reasonable that kv cache optimization would head towards this direction. 

Another more interesting and more personal reason is that I feel like studying how to chunk kv-cache is closely related to the larger problems of solving LLM memory / long context reasoning. LLMs operate in very fine-grained tokens (each word has multiple tokens). I think human brains operate in a much higher-level coarse grain space. So I think the key to solving long-horizon is just to have the model learn to chunk its tokens so it reason at a higher-level of abstraction. And the different research hand-designing how to chunk kv-cache feels a lot like before AlexNet was a thing and people were hand-designing ways to extract visual features. I think we will eventually reach a stage where chunking kv-cache is just something the model learns from training instead of an algorithm we need to hand design. 

2. Proposal / Method

I started looking at the problem from a very math-oriented angle: We want to minimize 
$$
E_{q} \left[||\sum_j v_j e^{k_j \cdot q} -  \sum_j v_j a_j e^{k^* \cdot q + c} ||\right]
$$
where $a_j$ are the weighing coefficients of the merged value $v^* = \sum_j v_j a_j$ where $c$ is some constant bias, which in practice can be $c=\ln N$ the number of tokens in the set. 

And $k^*$ is the merged key. There is another condition: we also want to minimize
$$
E_{q} \left[||\sum_j e^{k_j \cdot q} -  N e^{k^* \cdot q} ||\right]
$$
so that the normalization factor in attention doesn't change a lot (otherwise it would mess up attention to other tokens). Taking derivative with respect $k^*$ gives us the condition, 

$$
a_j E[e^{2k^* \cdot q}] = E[e^{2k^* \cdot q} e^{(k_j - k^*) \cdot q} / N] 
$$

Making the reasonable assumption that $k_j - k^*$ is orthogonal to $k^*$ we get that $a_j = \frac{1}{N} E[e^{(k_j - k^*)\cdot q}]$, which we can approximate