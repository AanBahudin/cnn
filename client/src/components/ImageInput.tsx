import { useFormStatus } from '@/context/FormContext'
import { Button } from './ui/button'
import {Loader2} from 'lucide-react'
import { store } from '@/store'
import { setImage } from '@/cart/globalSlice'
import { useSelector } from 'react-redux'

const ImageInput = () => {

  const {isLoading} = useFormStatus()
  const {selectedImg} = useSelector((state:any) => state.globalState)

  const handleImg = (event: any) => {
    const file = event.target.files[0]
    if (file) {
      const image = URL.createObjectURL(file)
      store.dispatch(setImage(image))
    }
  }

  return (
    <>
      {selectedImg ? (
        <img src={selectedImg} alt='motif kain' className="w-full object-contain h-[400px] mx-auto rounded-xl border flex items-center justify-center" />
      ) : (
        <div className="w-full h-[400px] mx-auto rounded-xl border flex items-center justify-center" />
      )}
        <input onChange={handleImg} type="file" accept="image/*" name="motifKain" id="motifKain" className="hidden" />
        <input type="hidden" value={selectedImg} name='motifKain' id='motifKain' />

        <section className='w-full flex items-center gap-x-4'>
          <label htmlFor='motifKain' className={`flex-1 mt-4 bg-secondary-foreground dark:bg-secondary text-white py-2 rounded-md text-center text-sm ${isLoading ? 'hidden' : null}`}>Upload Gambar</label>
          <Button disabled={isLoading} type='submit' variant='default' className={`${!selectedImg ? 'hidden' : ''} flex-1 mt-4 py-2`}>
            {isLoading ? (
              <span className='flex gap-x-2 items-center'>
                <Loader2 className='w-8 h-8 animate-spin ease-in-out stroke-white' />
                Mengirim
              </span>
            ) : 'Kirim Gambar'}
          </Button>
        </section>
    </>
  )
}

export default ImageInput