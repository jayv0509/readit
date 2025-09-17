
def translate_text_sync(text,src,dest) -> str:
    import asyncio
    from googletrans import Translator

    async def _translate():
        translator = Translator()
        result = await translator.translate(text, src=src, dest=dest)
        return result.text

    return asyncio.run(_translate())

# usage


