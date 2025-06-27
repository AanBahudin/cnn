import FormContainer from '@/globals/FormContainer'
import Container from '@/components/Container'
import ImageInput from '@/components/ImageInput'
import { processingData } from '@/actions/predictActions'
import ProcessingDataContainer from '@/components/ProcessingDataContainer'

const ProcessingData = () => {
  return (
    <FormContainer action={processingData}>
        <section className="min-h-[100vh] w-[100vw] flex items-center">
            <Container className="w-[80%] border h-[80%] flex justify-between items-center gap-x-10">
                <div className="w-1/2 flex items-center justify-center flex-col">
                <ImageInput />
                </div>
                <ProcessingDataContainer />
            </Container>
      </section>
    </FormContainer>
  )
}

export default ProcessingData