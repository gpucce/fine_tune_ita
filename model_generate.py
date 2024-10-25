from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "sapienzanlp/minerva_350m_summarization_fanpage"


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    tokenizer.pad_token_id = tokenizer.eos_token_id

    prompt = """###Text: La Paris Fashion Week volge al termine ma sono ancora molte le Maison di fama internazionale che stanno presentando le collezioni per la Primavera/Estate 2019. Ieri sera è stato il turno di uno dei nomi più attesi, Chanel, che per l'ennesima volta ha organizzato uno show scenografico al Grand Palais, trasformato per l'occasione in una vera e propria spiaggia con tanto di sabbia, bagnasciuga e onde. Karl Lagerfeld è stato capace di dare vita a qualcosa di incredibile, portando un pizzico di originalità alla Settimana della Moda della capitale francese. Chanel ha presentato la collezione Primavera/Estate 2019 durante la Paris Fashion Week, allestendo il Grand Palais in modo incredibile. La location che ospita tutte le sfilate della Maison è stata infatti trasformata in una spiaggia con tanto di onde, mare e bagnasciuga, sul quale le modelle hanno sfilato senza scarpe con in mano delle ciabatte e indosso degli abiti a tema marino. Camicie decorate con ombrello, maxi cappelli di paglia, borse spugna, sono solo alcune delle novità proposte sul catwalk da Karl Lagerfeld, che non ha rinunciato ad alcuni tratti distintivi del suo stile come i tailleur in tweed e l'iconico bouclé, declinati in colori accesi ispirati a quelli dei gelati, giacche dalle spalle importanti e squadrate e logo in bella vista sia sulle camicie che sulle borse. Sono proprio queste ultime a essere diventate protagoniste della sfilata: secchielli, marsupi, pochette, tracolle e addirittura delle "doppie borse", sono tutte perfette per aggiungere un tocco casual a ogni tipo di look estivo.
    
    ### Summary:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs["attention_mask"], max_length=1024)

    prompt_length = inputs['input_ids'].shape[1]

    generated_ids = outputs[0][prompt_length:]

    generated_answer = tokenizer.decode(generated_ids)

    print(f"generated ids {generated_ids}")
    print(f"generated answer {generated_answer}")
    

if __name__ == "__main__":
    main()